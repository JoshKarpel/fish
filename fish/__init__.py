import os as _os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
_os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from .io import read, save, load, cached_read, make_movie
from .bgnd import (
    train_background_subtractor,
    apply_background_subtraction,
    background_via_min,
    subtract_background,
)
from .clustering import (
    frame_to_chunks,
    make_all_batches,
    make_all_vectors_from_frames,
    make_batches_of_vectors,
    make_labels_over_time_stackplot,
    make_vectors_from_frames,
    normalized_pca_transform,
)
from .vectorize import sorted_ravel, sorted_diff, sorted_ds
from .edges import (
    get_edges,
    draw_bounding_rectangles,
    draw_object_tracks,
    detect_objects,
    Object,
    ObjectTrack,
    ObjectTracker,
)
from .dish import (
    clean_frame_for_hough_transform,
    remove_components_below_cutoff_area,
    find_circles_via_hough_transform,
    decide_dish,
    draw_circles,
    find_dish,
    area_ratio,
)
from .pipette import find_last_pipette_frame
from .flow import optical_flow, average_velocity_per_frame, total_velocity_per_frame
from .blobs import (
    BrightnessBlob,
    VelocityBlob,
    find_brightness_blobs,
    find_velocity_blobs,
save_blobs,
load_blobs
)
from .hand import HandCounted, load_hand_counted_data
from .domains import (
    domain,
    domain_circle,
    iter_domain_indices,
    rotate_domain_xy,
    interpolate_frame,
    evaluate_interpolation,
)
from .colors import (
    GREEN,
    YELLOW,
    RED,
    BLUE,
    WHITE,
    CYAN,
    MAGENTA,
    BLACK,
    fractions,
    convert_colorspace,
)
from .figs import (
    show_image,
    color_labels,
    overlay_image,
    draw_text,
    draw_arrow,
    draw_circle,
    draw_ellipse,
    draw_rectangle,
)
from .utils import (
    BlockTimer,
    chunk,
    window,
    distance_between,
    moving_average,
    apply_mask,
)
