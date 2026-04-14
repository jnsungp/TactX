import asyncio
import http
import logging
import time
import traceback

import numpy as np
import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _decode_raw_numpy(obj):
    """Decode raw msgpack_numpy dict format to numpy array.

    When msgpack_numpy versions differ between client/server, arrays may arrive
    as dicts with keys: {b'nd', b'type', b'kind', b'shape', b'data'}
    """
    if isinstance(obj, dict) and (b'nd' in obj or 'nd' in obj):
        type_key = b'type' if b'type' in obj else 'type'
        shape_key = b'shape' if b'shape' in obj else 'shape'
        data_key = b'data' if b'data' in obj else 'data'

        dtype_str = obj[type_key]
        if isinstance(dtype_str, bytes):
            dtype_str = dtype_str.decode()
        return np.frombuffer(obj[data_key], dtype=np.dtype(dtype_str)).reshape(tuple(obj[shape_key])).copy()
    return obj


def _decode_obs_recursive(obj):
    """Recursively decode any raw msgpack_numpy dicts."""
    if isinstance(obj, dict):
        if b'nd' in obj or 'nd' in obj:
            return _decode_raw_numpy(obj)
        return {k: _decode_obs_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_obs_recursive(item) for item in obj]
    return obj


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        norm_stats: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._norm_stats = norm_stats
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        debug_printed = False
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                # Handle raw msgpack_numpy dicts that weren't auto-decoded
                obs = _decode_obs_recursive(obs)

                # Debug: print received observation on first message
                if not debug_printed:
                    debug_printed = True
                    logger.info(f"[DEBUG] Received observation:")
                    logger.info(f"  obs keys: {obs.keys()}")
                    logger.info(f"  qpos type: {type(obs['qpos'])}")
                    if isinstance(obs['qpos'], dict):
                        logger.info(f"  qpos dict keys: {obs['qpos'].keys()}")
                    else:
                        logger.info(f"  qpos shape: {obs['qpos'].shape}, dtype: {obs['qpos'].dtype}")

                infer_time = time.monotonic()
                obs['qpos'] = (obs['qpos'] - self._norm_stats["qpos_mean"]) / (self._norm_stats["qpos_std"] + 1e-6)
                # Pass tactile data if present in observation
                tactile_images = obs.get('tactile_images', None)
                tactile_arrays = obs.get('tactile_arrays', None)
                # Normalize tactile arrays if stats are available
                if tactile_arrays is not None and self._norm_stats.get("tactile_mean") is not None:
                    tactile_arrays = (tactile_arrays - self._norm_stats["tactile_mean"]) / (self._norm_stats["tactile_std"] + 1e-6)
                    tactile_arrays = np.clip(tactile_arrays, -5.0, 5.0)
                action = self._policy(obs['image'], obs['qpos'], obs['cond_dict'], tactile_images, tactile_arrays)
                action['action'] = action['action'] * (self._norm_stats["action_std"] + 1e-6) + self._norm_stats["action_mean"]
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None