import http.client
import ssl

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cyberarena.site_audit import audit_site


router = APIRouter(
    prefix="/siteguard",
    tags=["SiteGuard"],
)


class SiteAuditRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    authorized_host: str = Field(min_length=1, max_length=253)
    scope_confirmed: bool = False
    allow_private_lab: bool = False


@router.post("/audit")
def run_site_audit(request: SiteAuditRequest):
    if not request.scope_confirmed:
        raise HTTPException(
            status_code=403,
            detail="Explicit authorization confirmation is required.",
        )

    try:
        result = audit_site(
            request.url,
            request.authorized_host,
            allow_private_lab=request.allow_private_lab,
        )
        return result.to_dict()

    except (
        ValueError,
        OSError,
        TimeoutError,
        ssl.SSLError,
        http.client.HTTPException,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
