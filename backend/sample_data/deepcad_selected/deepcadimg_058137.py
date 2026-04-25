import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.296875, -0.0390625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.2963815789473684, 0.0).lineTo(0.2963815789473684, 0.41406249999999994).lineTo(0.0, 0.41406249999999994).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.1481907894736842, 0.20921052631578946).circle(0.034868421052631576)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(-0.03125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.0390625, -0.75), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop2=wp_sketch1.moveTo(0.41842105263157897, 0.0).lineTo(0.41842105263157897, 0.75).lineTo(0.0, 0.75).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop2).extrude(0.03125)
solid=solid.union(solid1)