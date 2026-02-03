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


def create_angle_geometry(h, b, t, r_root, r_toe, section_id="LU"):
    """
    Creates an L-section where:
    - Heel (0,0) and Outer Tips are sharp.
    - Root (r_root) is concave.
    - Inner Toes (r_toe) are convex.
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
    points.append((b, t - r_toe))     # Go up the flat end face

    # 3. Bottom Leg - Inner Toe Fillet (Convex)
    # Rounds from the end face to the inner top face of the bottom leg
    # Center: (b - r_toe, t - r_toe)
    # Arc: 0 rad -> 90 rad (pi/2)
    points.extend(get_arc((b - r_toe, t - r_toe), 0, 0.5 * np.pi, r_toe))

    # 4. Root Fillet (Concave)
    # Connects the inner bottom face to the inner vertical face
    # Center: (t + r_root, t + r_root)
    # Arc: 270 rad (1.5pi) -> 180 rad (pi) [Clockwise logic, so we reverse]
    # We use linspace 1.5pi -> pi directly
    points.extend(get_arc((t + r_root, t + r_root), 1.5 * np.pi, np.pi, r_root))

    # 5. Top Leg - Inner Toe Fillet (Convex)
    # Rounds from the inner vertical face to the top end face
    # Center: (t - r_toe, h - r_toe)
    # Arc: 0 rad -> 90 rad (pi/2)
    points.extend(get_arc((t - r_toe, h - r_toe), 0, 0.5 * np.pi, r_toe))

    # 6. Top Leg - Tip & Outer Edge (Sharp)
    points.append((0, h))          # Top-left corner

    # Close the loop
    points.append((0, 0))

    return Polygon(points)


def create_i_section_geometry(h, b, tf, tw, r_root, section_id="I-Section"):
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
    # Center: (Web_Right + r_root, Flange_Bot + r_root)
    # Arc: 270 deg -> 180 deg
    center_br = (x_web_right + r_root, y_flange_bot_top + r_root)
    points.extend(get_arc(center_br, 1.5 * np.pi, np.pi, r_root))

    # 3. Web (Right Side, moving Up)
    # The arc ends exactly at the web face, so we just continue...

    # 4. Top-Right Root Fillet (Connects Web to Top Flange)
    # Center: (Web_Right + r_root, Flange_Top_Bot - r_root)
    # Arc: 180 deg -> 90 deg
    center_tr = (x_web_right + r_root, y_flange_top_bot - r_root)
    points.extend(get_arc(center_tr, np.pi, 0.5 * np.pi, r_root))

    # 5. Top Flange (Right to Left)
    points.append((x_right, y_flange_top_bot))  # Top-Right Inner Tip
    points.append((x_right, y_top))            # Top-Right Outer
    points.append((x_left, y_top))             # Top-Left Outer
    points.append((x_left, y_flange_top_bot))  # Top-Left Inner Tip

    # 6. Top-Left Root Fillet (Connects Top Flange to Web)
    # Center: (Web_Left - r_root, Flange_Top_Bot - r_root)
    # Arc: 90 deg -> 0 deg
    center_tl = (x_web_left - r_root, y_flange_top_bot - r_root)
    points.extend(get_arc(center_tl, 0.5 * np.pi, 0.0, r_root))

    # 7. Web (Left Side, moving Down)

    # 8. Bottom-Left Root Fillet (Connects Web to Bottom Flange)
    # Center: (Web_Left - r_root, Flange_Bot + r_root)
    # Arc: 0 deg -> -90 deg
    center_bl = (x_web_left - r_root, y_flange_bot_top + r_root)
    points.extend(get_arc(center_bl, 0.0, -0.5 * np.pi, r_root))

    # Close the loop
    points.append((x_left, y_flange_bot_top))  # Bottom-Left Inner Tip
    # Back to Origin (Redundant but safe)
    points.append((x_left, y_bottom))

    return Polygon(points)



def create_ipn_section_geometry(h, b, tf, tw, r_root, r_toe, section_id="IPN"):
    """
    Creates a Tapered I-Beam (IPN/INP).
    Standard slope is 14% (approx 8 degrees).
    tf is the MEAN flange thickness (measured at b/4 from the edge, i.e., x=b/4 from centroid).
    """
    res = 16
    slope = 0.14
    angle = np.arctan(slope) # ~0.139 radians (8 degrees)

    # --- 1. Coordinate Setup (Top-Right Quadrant) ---
    # We work relative to the Centroid (0,0) first, then mirror.
    # Top Edge: y = h/2
    # Right Edge: x = b/2
    
    # Inner Flange Slope Line Equation:
    # At x = b/4 (quarter width from center), the thickness is tf.
    # So the y-coordinate of the inner surface at x=b/4 is (h/2 - tf).
    # Since flanges get THINNER at the tip, the inner surface rises as x increases.
    # Equation: y_inner(x) = (h/2 - tf) + slope * (x - b/4)
    
    def get_y_slope(x):
        return (h/2 - tf) + slope * (x - b/4)

    # --- 2. Fillet Center Solvers ---
    
    def get_root_center(radius):
        """
        Root fillet is CONCAVE (adds material).
        Tangent to Vertical Web (x = tw/2) and Sloped Flange.
        Center is in the VOID (Right of web, Below slope).
        """
        cx = tw/2 + radius
        
        # Calculate vertical distance from the slope line to the center
        # The radius is the perpendicular distance.
        # Vertical distance dy = radius / cos(angle)
        cy = get_y_slope(cx) - (radius / np.cos(angle))
        return cx, cy

    def get_toe_center(radius):
        """
        Toe fillet is CONVEX (removes material).
        Tangent to Vertical Edge (x = b/2) and Sloped Flange.
        Center is in the MATERIAL (Left of edge, Above slope).
        """
        cx = b/2 - radius
        
        # Vertical distance dy = radius / cos(angle)
        # Since we are ABOVE the line (inside material), we add dy.
        cy = get_y_slope(cx) + (radius / np.cos(angle))
        return cx, cy

    # --- 3. Path Generation (Top-Right) ---
    points = []
    
    # A. Web Face (Start at bottom of quadrant, go up)
    points.append((tw/2, 0))
    
    # B. Root Fillet (Concave)
    c_root = get_root_center(r_root)
    # Start Angle: 180 deg (Pi) -> Tangent to web
    # End Angle: Tangent to slope. Normal to slope points down-right (-90+alpha).
    # Vector from center to tangent points UP-LEFT (90+alpha).
    ang_start_root = np.pi
    ang_end_root = np.pi/2 + angle
    
    rads_root = np.linspace(ang_start_root, ang_end_root, res)
    for a in rads_root:
        points.append((c_root[0] + r_root*np.cos(a), c_root[1] + r_root*np.sin(a)))
        
    # C. Toe Fillet (Convex) - or straight line if r_toe is 0/NaN
    if r_toe > 0 and not np.isnan(r_toe):
        c_toe = get_toe_center(r_toe)
        # Start Angle: Tangent to slope. Vector from center to tangent points DOWN-RIGHT.
        # Normal to slope is (-90+alpha). 
        ang_start_toe = -np.pi/2 + angle
        # End Angle: 0 deg -> Tangent to vertical right edge
        ang_end_toe = 0.0
        
        rads_toe = np.linspace(ang_start_toe, ang_end_toe, res)
        for a in rads_toe:
            points.append((c_toe[0] + r_toe*np.cos(a), c_toe[1] + r_toe*np.sin(a)))
    else:
        # Extend slope line to the edge
        # Intersection of y_slope(x) and x = b/2
        y_tip = get_y_slope(b/2)
        points.append((b/2, y_tip))

    # D. Vertical Edge (Up to top corner)
    points.append((b/2, h/2))
    
    # E. Top Face (Left to Y-axis)
    points.append((0, h/2))
    
    # --- 4. Symmetry & Shift ---
    # Mirror Top-Right to Top-Left
    poly_tr = points
    poly_tl = [(-x, y) for x, y in poly_tr][::-1]
    
    # Combine to make Top Half
    top_half = poly_tr + poly_tl
    
    # Mirror Top Half to Bottom Half
    bottom_half = [(x, -y) for x, y in top_half][::-1]
    
    full_points = top_half + bottom_half
    
    # Shift so Bottom-Left is at (0,0) to match other profiles
    shifted_points = [(x + b/2, y + h/2) for x, y in full_points]
    
    return Polygon(shifted_points)


def create_upn_section_geometry(h, b, tf, tw, r_root, r_toe, slope=None, section_id="UPN"):
    """
    Creates a Tapered Channel (UPN/UE).
    If slope is None, uses height-based UPN logic (8% for h<=300, else 5%).
    If slope is provided (e.g., 0.08), uses that fixed value.
    tf is defined at the midpoint of the flange (x = b/2).
    """
    res = 32

    # 1. Determine Slope (Auto-detect or Manual)
    if slope is None:
        current_slope = 0.08 if h <= 300 else 0.05
    else:
        current_slope = slope

    angle = np.arctan(current_slope)

    # 2. Slope Equation (tf at b/2)
    def get_y_slope(x):
        return (h/2 - tf) + current_slope * (x - b/2)

    # 3. Fillet Center Solvers
    def get_root_center(radius):
        cx = tw + radius
        cy = get_y_slope(cx) - (radius / np.cos(angle))
        return cx, cy

    def get_toe_center(radius):
        cx = b - radius
        cy = get_y_slope(cx) + (radius / np.cos(angle))
        return cx, cy

    # --- 4. Path Construction (Top Half) ---
    points_top = []
    points_top.append((0, h/2))  # Back of web
    points_top.append((b, h/2))  # Outer tip

    # Toe Fillet
    if r_toe > 0 and not np.isnan(r_toe):
        c_toe = get_toe_center(r_toe)
        angs = np.linspace(0, angle - np.pi/2, res)
        for a in angs:
            points_top.append(
                (c_toe[0] + r_toe*np.cos(a), c_toe[1] + r_toe*np.sin(a)))
    else:
        points_top.append((b, get_y_slope(b)))

    # Root Fillet
    c_root = get_root_center(r_root)
    angs = np.linspace(angle + np.pi/2, np.pi, res)
    for a in angs:
        points_top.append(
            (c_root[0] + r_root*np.cos(a), c_root[1] + r_root*np.sin(a)))

    points_top.append((tw, 0))  # Web face

    # --- 5. Mirror and Finish ---
    points_bottom = [(x, -y) for x, y in points_top][::-1]
    full_path = points_bottom + points_top
    final_points = [(x, y + h/2) for x, y in full_path]

    return Polygon(final_points)


def create_ue_section_geometry(h, b, tf, tw, r_root, r_toe, section_id="UE"):
    """
    Wrapper for UE channels using a fixed slope.
    """
    return create_upn_section_geometry(h, b, tf, tw, r_root, r_toe, slope=0.05, section_id=section_id)


def create_t_section_geometry(h, b, tf, tw, r_root, r_toe, r_web, section_id="T-Section"):
    """
    Creates a Tapered T-Section (T-profile) with corrected root fillets.
    - Top surface is flat.
    - Flange underside has 2% slope.
    - Web sides have 2% slope.
    - tf measured at b/4.
    - tw measured at h/2.
    """
    res = 16
    slope = 0.02
    alpha_f = np.arctan(slope)
    alpha_w = np.arctan(slope)

    # --- Line Equations ---
    def get_y_flange(x):
        return (h - tf) + slope * (x - b/4)

    def get_x_web(y):
        return tw/2 + slope * (y - h/2)

    # --- Arc Generator ---
    def get_arc(center, start_ang, end_ang, radius):
        angs = np.linspace(start_ang, end_ang, res)
        return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angs]

    points = []

    # --- Path Construction (Right Half, Counter-Clockwise) ---

    # A. Top Center & Corner
    points.append((0, h))
    points.append((b/2, h))

    # B. Flange Toe (r_toe)
    if r_toe > 0:
        cx = b/2 - r_toe
        dy = r_toe / np.cos(alpha_f)
        cy = get_y_flange(cx) + dy
        # From 0 deg (Vertical edge) to Normal-to-slope (-pi/2 + alpha)
        points.extend(get_arc((cx, cy), 0, -np.pi/2 + alpha_f, r_toe))
    else:
        points.append((b/2, get_y_flange(b/2)))

    # C. Root Fillet (r_root) - FIXED
    # 1. Find Center
    dist_f = r_root / np.cos(alpha_f)
    dist_w = r_root / np.cos(alpha_w)

    A_prime = (h - tf) - dist_f
    B_prime = (tw/2 + dist_w) - slope * h/2 - b/4

    cy_root = (A_prime + slope * B_prime) / (1 - slope**2)
    cx_root = (tw/2 + dist_w) + slope * (cy_root - h/2)

    # 2. Generate Arc (Concave)
    # Start Angle: Tangent to Flange. Normal is (pi/2 + alpha_f)
    # End Angle: Tangent to Web. Normal is (pi - alpha_w)
    ang_start_root = np.pi/2 + alpha_f
    ang_end_root = np.pi - alpha_w

    points.extend(get_arc((cx_root, cy_root),
                  ang_start_root, ang_end_root, r_root))

    # D. Web Tip (r_web)
    if r_web > 0:
        cy_web = r_web
        dx = r_web / np.cos(alpha_w)
        x_on_line = get_x_web(r_web)
        cx_web = x_on_line - dx
        # From Normal-to-web (-alpha_w) to -pi/2 (Down)
        points.extend(get_arc((cx_web, cy_web), -alpha_w, -np.pi/2, r_web))
        points.append((0, 0))
    else:
        points.append((get_x_web(0), 0))
        points.append((0, 0))

    # --- Mirror & Shift ---
    points_right = points
    points_left = [(-x, y) for x, y in points_right][::-1]
    full_points = points_left + points_right

    # Shift to Bottom-Left at (0,0)
    final_points = [(x + b/2, y) for x, y in full_points]

    return Polygon(final_points)


def create_u_section_geometry(h, b, tf, tw, r_root, section_id="U-Section"):
    """
    Creates a Parallel Flange Channel (UPE or UAP).
    Origin (0,0) is at the external heel (back of the web).
    """
    res = 16  # Resolution for fillet arcs

    def get_arc(center, start_angle, end_angle, radius):
        angles = np.linspace(start_angle, end_angle, res)
        return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]

    # Key X-coordinates
    x_back = 0
    x_tip = b
    x_inner_web = tw

    # Key Y-coordinates
    y_bottom = 0
    y_top = h
    y_inner_bot = tf
    y_inner_top = h - tf

    # --- Path Construction (Counter-Clockwise) ---
    points = []

    # 1. External Back and Bottom
    points.append((x_back, y_bottom))  # Bottom-Back heel
    points.append((x_tip, y_bottom))   # Bottom-Front tip
    points.append((x_tip, y_inner_bot))  # Bottom-Front inner corner

    # 2. Bottom Root Fillet (Connects Bottom Flange to Web)
    # Center: (tw + r_root, tf + r_root)
    # Arc: 270 deg -> 180 deg
    center_bot = (x_inner_web + r_root, y_inner_bot + r_root)
    points.extend(get_arc(center_bot, 1.5 * np.pi, np.pi, r_root))

    # 3. Top Root Fillet (Connects Web to Top Flange)
    # Center: (tw + r_root, (h - tf) - r_root)
    # Arc: 180 deg -> 90 deg
    center_top = (x_inner_web + r_root, y_inner_top - r_root)
    points.extend(get_arc(center_top, np.pi, 0.5 * np.pi, r_root))

    # 4. Top Flange and Back
    points.append((x_tip, y_inner_top))  # Top-Front inner corner
    points.append((x_tip, y_top))       # Top-Front tip
    points.append((x_back, y_top))      # Top-Back heel

    # Close the loop
    points.append((x_back, y_bottom))

    return Polygon(points)
