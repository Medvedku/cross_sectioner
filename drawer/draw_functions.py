import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import matplotlib.pyplot as plt


def create_rhs_geometry(h, b, t, r_out, r_in, section_id="RHS"):
    """
    Creates a Shapely geometry for a Rectangular Hollow Section.
    """
    def rounded_rect(width, height, radius):
        # Handle cases where radius might be 0 to avoid buffer errors
        if radius <= 0:
            return box(-width/2, -height/2, width/2, height/2)

        core = box(-(width/2 - radius), -(height/2 - radius),
                   (width/2 - radius), (height/2 - radius))
        return core.buffer(radius, resolution=32)

    # Construct shapes
    outer_poly = rounded_rect(b, h, r_out)
    inner_poly = rounded_rect(b - 2*t, h - 2*t, r_in)

    # Subtract to get hollow section
    rhs_section = outer_poly.difference(inner_poly)
    return rhs_section


def create_chs_geometry(d, t, section_id="CHS"):
    """
    Creates a Shapely geometry for a Circular Hollow Section.
    """
    r_out = d / 2
    r_in = r_out - t

    # Create the hollow section
    outer_circle = Point(0, 0).buffer(r_out, resolution=128)
    inner_circle = Point(0, 0).buffer(r_in, resolution=128)

    return outer_circle.difference(inner_circle)


def create_shs_geometry(a, t, r_out, r_in, section_id="SHS"):
    """
    Creates a Shapely geometry for a Square Hollow Section.
    Uses create_rhs_geometry logic where h = b = a.
    """
    return create_rhs_geometry(h=a, b=a, t=t, r_out=r_out, r_in=r_in, section_id=section_id)


def create_angle_geometry(h, b, t, r1, r2, section_id="LU"):
    """
    Creates an L-section where:
    - Heel (0,0) and Outer Tips are sharp.
    - Root (r1) is concave.
    - Inner Toes (r2) are convex.
    """
    res = 16  # Resolution for arcs

    def get_arc(center, start_angle, end_angle, radius):
        angles = np.linspace(start_angle, end_angle, res)
        return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]

    # --- Path Construction (Counter-Clockwise) ---
    points = []

    # 1. Heel (Sharp Origin)
    points.append((0, 0))

    # 2. Bottom Leg - Outer Edge & Tip (Sharp)
    points.append((b, 0))          # Bottom-right corner
    points.append((b, t - r2))     # Go up the flat end face

    # 3. Bottom Leg - Inner Toe Fillet (Convex)
    # Rounds from the end face to the inner top face of the bottom leg
    # Center: (b - r2, t - r2)
    # Arc: 0 rad -> 90 rad (pi/2)
    points.extend(get_arc((b - r2, t - r2), 0, 0.5 * np.pi, r2))

    # 4. Root Fillet (Concave)
    # Connects the inner bottom face to the inner vertical face
    # Center: (t + r1, t + r1)
    # Arc: 270 rad (1.5pi) -> 180 rad (pi) [Clockwise logic, so we reverse]
    # We use linspace 1.5pi -> pi directly
    points.extend(get_arc((t + r1, t + r1), 1.5 * np.pi, np.pi, r1))

    # 5. Top Leg - Inner Toe Fillet (Convex)
    # Rounds from the inner vertical face to the top end face
    # Center: (t - r2, h - r2)
    # Arc: 0 rad -> 90 rad (pi/2)
    points.extend(get_arc((t - r2, h - r2), 0, 0.5 * np.pi, r2))

    # 6. Top Leg - Tip & Outer Edge (Sharp)
    points.append((0, h))          # Top-left corner

    # Close the loop
    points.append((0, 0))

    return Polygon(points)


def create_i_section_geometry(h, b, tf, tw, r1, section_id="I-Section"):
    """
    Creates a standard I-beam or H-beam profile (Parallel Flanges).
    Covers IPE, HEA, HEB, HEM.
    Origin (0,0) is at the bottom-left corner of the bounding box.
    """
    res = 16  # Resolution for each fillet arc

    def get_arc(center, start_angle, end_angle, radius):
        # Generates points for an arc between two angles
        angles = np.linspace(start_angle, end_angle, res)
        return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]

    # Pre-calculate key X-coordinates
    x_left = 0
    x_right = b
    x_web_left = (b / 2) - (tw / 2)
    x_web_right = (b / 2) + (tw / 2)

    # Pre-calculate key Y-coordinates
    y_bottom = 0
    y_top = h
    y_flange_bot_top = tf
    y_flange_top_bot = h - tf

    # --- Path Construction (Counter-Clockwise) ---
    points = []

    # 1. Bottom Flange (Left to Right)
    points.append((x_left, y_bottom))       # Bottom-Left Outer
    points.append((x_right, y_bottom))      # Bottom-Right Outer
    points.append((x_right, y_flange_bot_top))  # Bottom-Right Inner Tip

    # 2. Bottom-Right Root Fillet (Connects Bottom Flange to Web)
    # Center: (Web_Right + r1, Flange_Bot + r1)
    # Arc: 270 deg -> 180 deg
    center_br = (x_web_right + r1, y_flange_bot_top + r1)
    points.extend(get_arc(center_br, 1.5 * np.pi, np.pi, r1))

    # 3. Web (Right Side, moving Up)
    # The arc ends exactly at the web face, so we just continue...

    # 4. Top-Right Root Fillet (Connects Web to Top Flange)
    # Center: (Web_Right + r1, Flange_Top_Bot - r1)
    # Arc: 180 deg -> 90 deg
    center_tr = (x_web_right + r1, y_flange_top_bot - r1)
    points.extend(get_arc(center_tr, np.pi, 0.5 * np.pi, r1))

    # 5. Top Flange (Right to Left)
    points.append((x_right, y_flange_top_bot))  # Top-Right Inner Tip
    points.append((x_right, y_top))            # Top-Right Outer
    points.append((x_left, y_top))             # Top-Left Outer
    points.append((x_left, y_flange_top_bot))  # Top-Left Inner Tip

    # 6. Top-Left Root Fillet (Connects Top Flange to Web)
    # Center: (Web_Left - r1, Flange_Top_Bot - r1)
    # Arc: 90 deg -> 0 deg
    center_tl = (x_web_left - r1, y_flange_top_bot - r1)
    points.extend(get_arc(center_tl, 0.5 * np.pi, 0.0, r1))

    # 7. Web (Left Side, moving Down)

    # 8. Bottom-Left Root Fillet (Connects Web to Bottom Flange)
    # Center: (Web_Left - r1, Flange_Bot + r1)
    # Arc: 0 deg -> -90 deg
    center_bl = (x_web_left - r1, y_flange_bot_top + r1)
    points.extend(get_arc(center_bl, 0.0, -0.5 * np.pi, r1))

    # Close the loop
    points.append((x_left, y_flange_bot_top))  # Bottom-Left Inner Tip
    # Back to Origin (Redundant but safe)
    points.append((x_left, y_bottom))

    return Polygon(points)
