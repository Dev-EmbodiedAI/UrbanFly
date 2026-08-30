"""Fast integration smoke test for the UrbanFly CityGS digital twin service."""

from __future__ import annotations

import asyncio
import json

import aiohttp


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8765/") as response:
            html = await response.text()
            assert response.status == 200
            assert "UrbanFly" in html

        config_url = (
            "http://localhost:8765/data/citygs_visualization/Residence/"
            "Residence_browser_config.json"
        )
        async with session.get(config_url) as response:
            assert response.status == 200
            viewer_config = await response.json()
            assert viewer_config["sceneName"] == "Residence"

        asset_url = (
            "http://localhost:8765/data/citygs_visualization/assets/"
            + viewer_config["modelAssetName"]
        )
        async with session.head(asset_url) as response:
            assert response.status == 200
            assert int(response.headers["Content-Length"]) > 20_000_000
            assert response.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
            assert response.headers.get("Cross-Origin-Embedder-Policy") == "require-corp"

        collision_root = (
            "http://localhost:8765/data/citygs_collision/Residence/"
        )
        async with session.get(collision_root + "collision_geometry.json") as response:
            assert response.status == 200
            collision_metadata = await response.json()
            assert collision_metadata["buildings"]["watertight"] is True
            assert collision_metadata["global_esdf"]["resolution_m"] == 1.0
            assert collision_metadata["local_esdf"]["resolution_m"] == 0.25

        async with session.head(collision_root + "city_collision.glb") as response:
            assert response.status == 200
            assert int(response.headers["Content-Length"]) > 10_000_000

        async with session.ws_connect("http://localhost:8765/ws") as socket:
            initial = [json.loads((await socket.receive()).data) for _ in range(2)]
            scenario_message = next(item for item in initial if item["type"] == "scenario_list")
            scenarios = scenario_message["payload"]
            assert scenarios
            scenario_name = scenarios[0]["name"]

            await socket.send_json({"type": "select_scenario", "payload": {"name": scenario_name}})
            started = None
            for _ in range(30):
                candidate = json.loads((await socket.receive()).data)
                if candidate["type"] == "scenario_start":
                    started = candidate
                    break
            assert started is not None
            assert started["type"] == "scenario_start"
            assert started["payload"]["bounds"]["size"][0] == 500.0

            await socket.send_json({"type": "control", "payload": {"action": "play"}})
            for _ in range(20):
                message = json.loads((await socket.receive()).data)
                if message["type"] == "sim_state":
                    assert len(message["payload"]["drones"]) > 0
                    print(
                        json.dumps(
                            {
                                "scene": "Residence CityGS",
                                "operation_extent_m": 500,
                                "scenario": scenario_name,
                                "drones": len(message["payload"]["drones"]),
                                "tasks": len(message["payload"]["tasks"]),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return
            raise AssertionError("No sim_state received")


if __name__ == "__main__":
    asyncio.run(main())
