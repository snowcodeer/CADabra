import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.75, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.5, 0.0).lineTo(1.5, 1.5).lineTo(0.0, 1.5).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.328125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.546875, -0.75, 0.1640625), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.07500000000000001, 0.0).circle(0.07500000000000001)
solid1=wp_sketch1.add(loop1).extrude(-0.625)
solid=solid.cut(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.3984375, -0.75, 0.1640625), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop2=wp_sketch2.moveTo(0.07500000000000001, 0.0).circle(0.07500000000000001)
solid2=wp_sketch2.add(loop2).extrude(-0.625)
solid=solid.cut(solid2)