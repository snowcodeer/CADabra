import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.46875, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.8984375, 0.0).lineTo(0.8984375, 0.5958059210526315).lineTo(0.7471217105263158, 0.5958059210526315).lineTo(0.7471217105263158, 0.3026315789473684).lineTo(0.1513157894736842, 0.3026315789473684).lineTo(0.1513157894736842, 0.5958059210526315).lineTo(0.0, 0.5958059210526315).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.75)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.3203125, -0.296875, 0.296875), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.6015625, 0.0).lineTo(0.6015625, 0.30394736842105263).lineTo(0.0, 0.30394736842105263).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.296875)
solid=solid.union(solid1)