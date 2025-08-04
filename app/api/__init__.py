# filepath: photo_product_api/photo_product_api/app/api/__init__.py
from fastapi import APIRouter

router = APIRouter()

from .photo_realistic_view import *
from .product_2d_to_3d import *
from .product_replacement import *