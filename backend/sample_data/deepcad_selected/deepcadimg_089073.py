import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5625, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.1875, 0.0).lineTo(0.1875, 0.46875).lineTo(0.0, 0.46875).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.375, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.37894736842105264, 0.0).lineTo(0.37894736842105264, 0.75).lineTo(0.0, 0.75).lineTo(0.0, 0.46578947368421053).lineTo(0.0, 0.0).close()
loop2=wp_sketch1.moveTo(0.18947368421052632, 0.18947368421052632).circle(0.09473684210526316)
loop3=wp_sketch1.moveTo(0.18947368421052632, 0.5605263157894737).circle(0.09473684210526316)
solid1=wp_sketch1.add(loop1).add(loop2).add(loop3).extrude(0.28125)
solid=solid.union(solid1)