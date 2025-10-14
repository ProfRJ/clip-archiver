import aiofiles
import asyncio
import json

from pathlib import Path

class Async_JSON:
    async def async_load_json(pathToLoad) -> dict:
        async with aiofiles.open(Path(pathToLoad), 'r') as file:
            contents = await file.read()
        return json.loads(contents)

    async def async_save_json(pathToSave, data) -> None:
        async with aiofiles.open(Path(pathToSave), 'w') as file:
            await file.write(json.dumps(data, ensure_ascii=False, indent=4))