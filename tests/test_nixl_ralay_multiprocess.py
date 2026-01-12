# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import multiprocessing
import pickle
import time
from queue import Empty
from typing import Any, Dict

import pytest
import torch

# Set multiprocessing start method to 'spawn' (required for CUDA)
if torch.cuda.is_available():
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Check if nixl_connect is available
try:
    from sglang_omni.relay.descriptor import Descriptor
    from sglang_omni.relay.nvxl.nixl_connect import Connector, RdmaMetadata
except ImportError:
    Connector = None
    Descriptor = None
    RdmaMetadata = None


def sender_process(
    config: Dict[str, Any],
    metadata_queue: multiprocessing.Queue,
    original_data_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
    done_event: multiprocessing.Event,
    num_transfers: int,
    test_data_size: int,
    results_dict: Dict[str, Any],
):
    """Sender process: creates data and sends via put_async."""

    async def run_sender():
        from sglang_omni.relay.nixl_ralay import NixlRalay

        connector = NixlRalay(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"

        try:
            put_times = []
            ready_event.set()

            for i in range(num_transfers):
                # Create test data
                original_tensor = torch.randn(
                    test_data_size, dtype=torch.float32, device=device
                )
                original_values = original_tensor.cpu().clone()

                send_descriptor = Descriptor(original_tensor)

                # Measure put time
                start = time.time()
                readable_op = await connector.put_async([send_descriptor])
                put_time = (time.time() - start) * 1000
                put_times.append(put_time)

                # Send metadata to receiver
                metadata = readable_op.metadata()
                try:
                    metadata_serialized = pickle.dumps(metadata)
                except Exception:
                    metadata_dict = (
                        metadata.model_dump()
                        if hasattr(metadata, "model_dump")
                        else metadata.dict()
                    )
                    metadata_serialized = pickle.dumps(metadata_dict)

                metadata_queue.put(
                    {
                        "idx": i,
                        "metadata": metadata_serialized,
                        "size": test_data_size,
                        "dtype": original_tensor.dtype,
                    }
                )

                original_data_queue.put(
                    {
                        "idx": i,
                        "data": pickle.dumps(original_values),
                    }
                )

                await readable_op.wait_for_completion()
                print(f"[Sender] Transfer {i + 1}/{num_transfers}: {put_time:.2f} ms")

            results_dict["sender_put_times"] = put_times
            metadata_queue.put(None)  # End signal
            done_event.wait(timeout=300)

        except Exception as e:
            results_dict["sender_error"] = str(e)
            import traceback

            traceback.print_exc()
        finally:
            connector.close()

    asyncio.run(run_sender())


def receiver_process(
    config: Dict[str, Any],
    metadata_queue: multiprocessing.Queue,
    original_data_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
    done_event: multiprocessing.Event,
    num_transfers: int,
    results_dict: Dict[str, Any],
):
    """Receiver process: receives data via get_async."""

    async def run_receiver():
        from sglang_omni.relay.nixl_ralay import NixlRalay
        from sglang_omni.relay.nvxl.nixl_connect import RdmaMetadata

        connector = NixlRalay(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"

        try:
            get_times = []
            ready_event.wait(timeout=30)

            count = 0
            while count < num_transfers:
                try:
                    item = metadata_queue.get(timeout=60)
                    if item is None:
                        break

                    # Deserialize metadata
                    metadata_obj = pickle.loads(item["metadata"])
                    metadata = (
                        RdmaMetadata(**metadata_obj)
                        if isinstance(metadata_obj, dict)
                        else metadata_obj
                    )

                    # Create receive buffer
                    buffer_tensor = torch.empty(
                        item["size"], dtype=item["dtype"], device=device
                    )
                    buffer_descriptor = Descriptor(buffer_tensor)

                    # Measure get time
                    start = time.time()
                    read_op = await connector.get_async(metadata, [buffer_descriptor])
                    if hasattr(read_op, "wait_for_completion"):
                        await read_op.wait_for_completion()
                    get_time = (time.time() - start) * 1000
                    get_times.append(get_time)

                    # Verify data
                    original_item = original_data_queue.get(timeout=10)
                    original_values = pickle.loads(original_item["data"])
                    received_values = buffer_tensor.cpu()

                    assert torch.allclose(
                        original_values, received_values, rtol=1e-5, atol=1e-5
                    ), f"Data mismatch in transfer {count + 1}"
                    assert not torch.allclose(
                        received_values, torch.zeros_like(received_values)
                    ), f"Received data is all zeros in transfer {count + 1}"

                    count += 1
                    print(
                        f"[Receiver] Transfer {count}/{num_transfers}: {get_time:.2f} ms - PASSED"
                    )

                except Empty:
                    print(f"[Receiver] Timeout waiting for data")
                    break
                except Exception as e:
                    results_dict["receiver_error"] = str(e)
                    import traceback

                    traceback.print_exc()
                    break

            results_dict["receiver_get_times"] = get_times
            done_event.set()

        except Exception as e:
            results_dict["receiver_error"] = str(e)
            import traceback

            traceback.print_exc()
        finally:
            connector.close()

    asyncio.run(run_receiver())


@pytest.mark.skipif(Connector is None, reason="nixl_connect not available")
@pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
def test_multiprocess_transfer_with_nixl_ralay():
    """Test data transfer between two processes using NixlRalay."""
    if torch.cuda.is_available():
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for this test")

    config_worker0 = {
        "host": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "device_name": "",
        "gpu_id": 0,
        "worker_id": "worker0",
    }

    config_worker1 = {
        "host": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "device_name": "",
        "gpu_id": 1 if torch.cuda.is_available() else 0,
        "worker_id": "worker1",
    }

    test_data_size = 100000  # 100K elements (~400 KB)
    num_transfers = 5

    # Create IPC objects
    metadata_queue = multiprocessing.Queue()
    original_data_queue = multiprocessing.Queue()
    ready_event = multiprocessing.Event()
    done_event = multiprocessing.Event()

    manager = multiprocessing.Manager()
    results_dict = manager.dict()

    print("\n" + "=" * 60)
    print("Multiprocess Transfer Test (NixlRalay)")
    print("=" * 60)
    print(
        f"Data size: {test_data_size} elements (~{test_data_size * 4 / (1024*1024):.2f} MB)"
    )
    print(f"Transfers: {num_transfers}")
    print(f"Sender: worker0 (GPU {config_worker0['gpu_id']})")
    print(f"Receiver: worker1 (GPU {config_worker1['gpu_id']})")
    print("-" * 60)

    sender = multiprocessing.Process(
        target=sender_process,
        args=(
            config_worker0,
            metadata_queue,
            original_data_queue,
            ready_event,
            done_event,
            num_transfers,
            test_data_size,
            results_dict,
        ),
    )

    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(
            config_worker1,
            metadata_queue,
            original_data_queue,
            ready_event,
            done_event,
            num_transfers,
            results_dict,
        ),
    )

    try:
        sender.start()
        receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        if sender.exitcode != 0:
            pytest.fail(f"Sender process exited with code {sender.exitcode}")
        if receiver.exitcode != 0:
            pytest.fail(f"Receiver process exited with code {receiver.exitcode}")

        if "sender_error" in results_dict:
            pytest.fail(f"Sender error: {results_dict['sender_error']}")
        if "receiver_error" in results_dict:
            pytest.fail(f"Receiver error: {results_dict['receiver_error']}")

        # Print statistics
        print("\n" + "=" * 60)
        print("Transfer Time Statistics")
        print("=" * 60)

        if "sender_put_times" in results_dict:
            times = list(results_dict["sender_put_times"])
            print(f"\n[Sender] Put times:")
            print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
            print(f"  Average: {sum(times)/len(times):.2f} ms")

        if "receiver_get_times" in results_dict:
            times = list(results_dict["receiver_get_times"])
            print(f"\n[Receiver] Get times:")
            print(f"  Range: {min(times):.2f} - {max(times):.2f} ms")
            print(f"  Average: {sum(times)/len(times):.2f} ms")

        print("\n" + "=" * 60)
        print("Test completed successfully")
        print("=" * 60)

    finally:
        if sender.is_alive():
            sender.terminate()
            sender.join(timeout=5)
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(timeout=5)


if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    pytest.main([__file__, "-v", "-s"])
