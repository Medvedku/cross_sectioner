import svgelements
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity


class CrossSection:
    def __init__(self, svg_path, num_samples=1000):
        self.svg_path = svg_path
        self.num_samples = num_samples

        self.svg_data = svgelements.SVG.parse(svg_path)
        self.header_width = self.svg_data.viewbox.width

        self.polygon = self._build_polygon()
        self.has_holes = len(self.polygon.interiors) > 0

        # 3. Structural Distances to Extreme Fibers
        minx, miny, maxx, maxy = self.polygon.bounds
        self.xLc = abs(minx)
        self.xRc = maxx
        self.yBc = abs(miny)
        self.yTc = maxy

    def _build_polygon(self):
        all_loops = []
        for element in self.svg_data.elements():
            if isinstance(element, svgelements.Path):
                # We split into subpaths (loops)
                try:
                    subs = list(element.as_subpaths())
                except AttributeError:
                    subs = [element]

                for sub in subs:
                    # Wrapping the subpath in a new Path object fixes the 'AttributeError'
                    # This ensures we can use the .point(t) method
                    temp_path = svgelements.Path(sub)

                    if temp_path.length() == 0:
                        continue

                    points = []
                    for t in np.linspace(0, 1, self.num_samples):
                        p = temp_path.point(t)
                        points.append((p.x, p.y))

                    if len(points) > 3:
                        all_loops.append(Polygon(points))

        if not all_loops:
            raise ValueError(f"No valid geometry in {self.svg_path}")

        # Sort: Largest = Shell
        all_loops.sort(key=lambda p: p.area, reverse=True)
        shell = all_loops[0].exterior
        holes = [p.exterior for p in all_loops[1:]]
        poly = Polygon(shell=shell, holes=holes)

        # Scale and Flip (Y-Up)
        raw_w = poly.bounds[2] - poly.bounds[0]
        scale = self.header_width / raw_w
        poly = affinity.scale(poly, xfact=scale, yfact=-scale, origin=(0, 0))

        # Center at (0,0)
        cx, cy = poly.centroid.x, poly.centroid.y
        return affinity.translate(poly, xoff=-cx, yoff=-cy)
    

    def calculate_inertia(self):
        """Calculates Ix, Iy, and Ixy for the section."""
        def _poly_inertia(ring):
            x, y = ring.coords.xy
            ix, iy, ixy = 0, 0, 0
            for i in range(len(x) - 1):
                # Common cross-product term
                a_i = x[i] * y[i+1] - x[i+1] * y[i]

                ix += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2) * a_i
                iy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2) * a_i
                ixy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2 *
                        x[i+1]*y[i+1] + x[i+1]*y[i]) * a_i

            # Use absolute values to be independent of vertex winding
            return abs(ix / 12.0), abs(iy / 12.0), ixy / 24.0

        # Start with the exterior shell
        Ix, Iy, Ixy_shell = _poly_inertia(self.polygon.exterior)

        # Subtract the holes
        for interior in self.polygon.interiors:
            h_ix, h_iy, h_ixy = _poly_inertia(interior)
            Ix -= h_ix
            Iy -= h_iy
            Ixy_shell -= h_ixy

        return Ix, Iy, Ixy_shell

    @property
    def area(self):
        return self.polygon.area

    @property
    def perimeter_exterior(self):
        return self.polygon.exterior.length

    @property
    def perimeter_total(self):
        return self.polygon.exterior.length + sum(h.length for h in self.polygon.interiors)

    @property
    def inertia(self):
        """Internal helper to calculate and return all inertia components."""
        # This calls your existing calculate_inertia method
        return self.calculate_inertia()

    @property
    def Ix(self):
        return self.inertia[0]

    @property
    def Iy(self):
        return self.inertia[1]

    @property
    def Ixy(self):
        return self.inertia[2]

    @property
    def principal_moments(self):
        """Returns (I1, I2) - the maximum and minimum moments of inertia."""
        ix, iy, ixy = self.Ix, self.Iy, self.Ixy
        avg_i = (ix + iy) / 2
        diff_i = np.sqrt(((ix - iy) / 2)**2 + ixy**2)
        return avg_i + diff_i, avg_i - diff_i

    @property
    def I1(self):
        """Strong axis moment of inertia (I_max)."""
        return self.principal_moments[0]

    @property
    def I2(self):
        """Weak axis moment of inertia (I_min)."""
        return self.principal_moments[1]

    @property
    def alpha(self):
        """Rotation angle from local x-axis to principal axis 1 in degrees."""
        ix, iy, ixy = self.Ix, self.Iy, self.Ixy
        # Negative result indicates clockwise rotation
        return 0.5 * np.degrees(np.arctan2(2 * ixy, iy - ix))

    @property
    def principal_section_moduli(self):
        """
        Returns (W1, W2) for the principal axes. 
        Note: This is an approximation using the bounding box 
        rotated to the principal angle.
        """
        # This is a complex calculation; for now, we can provide
        # local moduli which are standard for most checks.
        w_x = self.Ix / max(self.yTc, self.yBc)
        w_y = self.Iy / max(self.xRc, self.xLc)
        return w_x, w_y


    @property
    def J(self):
        """
        Universal Torsion Constant (St. Venant J).
        Uses Bredt's Theory for closed loops (RSH/CHS)
        and the (1/3)bt^3 approximation for open sections (L-angles).
        """
        if self.has_holes:
            # CLOSED SECTIONS (CHS, RHS, SHS)

            # 1. Calculate the area enclosed by the holes
            area_int = sum(Polygon(h).area for h in self.polygon.interiors)

            # 2. Calculate the total exterior area (Net Area + Hole Area)
            # This is safer than calling .area on a LinearRing
            area_ext = self.area + area_int

            # 3. Mean Enclosed Area (Am) and Mean Perimeter (Pm)
            area_mean = (area_ext + area_int) / 2

            peri_ext = self.polygon.exterior.length
            peri_int = sum(h.length for h in self.polygon.interiors)
            peri_mean = (peri_ext + peri_int) / 2

            # 4. Effective Thickness (t = Area / Mean Perimeter)
            t_eff = self.area / peri_mean

            # Bredt's Result: J = (4 * Am^2 * t) / Pm
            return (4 * (area_mean**2) * t_eff) / peri_mean

        else:
            # OPEN SECTIONS (L-angles, I-beams, etc.)
            avg_t = self.area / (self.perimeter_total / 2)
            return (1/3) * self.area * (avg_t**2)
