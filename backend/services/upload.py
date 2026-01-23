import os
import re
import asyncio
import hashlib
import httpx
import supabase
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from supabase.client import create_client
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import Annotated, List, Optional
import requests

# HTTP Status Codes
SUCCESS = 200
BAD_REQUEST = 400
INTERNAL_SERVER_ERROR = 500

# Error Messages
EMPTY_FILE = "File is empty"
FAILED_TO_DOWNLOAD_LINK_CONTENT = "Failed to download link content"

load_dotenv()

from utils.supabase_client import supabase_client as supabase

# supabase_url = os.getenv("SUPABASE_URL")
# supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase_bucket = os.getenv("SUPABASE_BUCKET")
# supabase = create_client(supabase_url, supabase_service_key)

app = FastAPI()

def _safe_name(name:str) -> str:
    name = os.path.basename(name or "upload.bin")
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name or "upload.bin"

def _make_path(user_id: str, project_id: str, kind: str, filename: str) -> str:
    user_id = _safe_name(user_id)
    project_id = _safe_name(project_id)
    filename = _safe_name(filename)
    return f"{user_id}/{project_id}/{kind}/{filename}"

def _upload_bytes(path: str, content: bytes, content_type:Optional[str]) -> str:
    return supabase.storage.from_(supabase_bucket).upload(
    path,
    content, 
    file_options={
        "content-type": content_type or "application/octet-stream",
        "upsert":"true"
     },
    )

def _public_url(path: str)-> str:
    return supabase.storage.from_(supabase_bucket).get_public_url(path)

class LinkInfo(BaseModel):
    url: HttpUrl

@app.post("/users/{user_id}/projects/{project_id}/files")
async def upload_files(user_id: str, project_id: str, file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=BAD_REQUEST, detail=EMPTY_FILE)

    path = _make_path(user_id, project_id, "files", file.filename)
    try:
        await run_in_threadpool(_upload_bytes, path, content, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=INTERNAL_SERVER_ERROR, detail=str(e))

    return {"path": path, "url": _public_url(path)}

@app.post("/users/{user_id}/projects/{project_id}/files/batch")
async def uplaod_files(user_id: str, project_id: str, files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=BAD_REQUEST, detail=EMPTY_FILE)
        path = _make_path(user_id, project_id, "files", file.filename)
        try:
            await run_in_threadpool(_upload_bytes, path, content, file.content_type)
        except Exception as e:
            raise HTTPException(status_code=INTERNAL_SERVER_ERROR, detail=str(e))
        results.append({"path": path, "url": _public_url(path)})
    return {"files": results}


@app.post("/users/{user_id}/projects/{project_id}/links")
async def upload_link(user_id: str, project_id: str, body: LinkInfo):
    url = str(body.url)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        r = await client.get(url)
    if r.status_code != SUCCESS or not r.content:
        raise HTTPException(status_code=BAD_REQUEST, detail=FAILED_TO_DOWNLOAD_LINK_CONTENT)

    base = url.split("?")[0].rstrip("/")
    name = base.split("/")[-1] if "/" in base else ""
    if not name:
        name = "link_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".bin"

    path = _make_path(user_id, project_id, "links", name)
    await run_in_threadpool(_upload_bytes, path, r.content, r.headers.get("content-type"))

    return {"path": path, "url": _public_url(path), "source_url": url}


@app.post("/users/{user_id}/projects/{project_id}/links/batch")
async def upload_links(user_id: str, project_id: str, links: List[HttpUrl]):
    results = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        for link in links:
            url = str(link)
            r = await client.get(url)
            if r.status_code != SUCCESS or not r.content:
                continue

            name = "link_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".bin"
            path = _make_path(user_id, project_id, "links", name)
            await run_in_threadpool(_upload_bytes, path, r.content, r.headers.get("content-type"))
            results.append({"path": path, "url": _public_url(path), "source_url": url})

    return {"links": results}

