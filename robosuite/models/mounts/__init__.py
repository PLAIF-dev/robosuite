from .mount_model import MountModel
from .mount_factory import mount_factory

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount


MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "NullMount": NullMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()
