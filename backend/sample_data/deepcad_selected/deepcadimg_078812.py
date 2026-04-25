import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.203125, 0.546875), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.1368421052631579, 0.0).threePointArc((0.203125, 0.06628289473684211), (0.2694078947368421, 0.0)).lineTo(0.40625, 0.0).threePointArc((0.203125, 0.203125), (0.0, 0.0)).close()
solid0=wp_sketch0.add(loop0).extrude(0.1328125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.203125, 0.0), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.0, 0.0).threePointArc((0.20435855263157893, -0.20435855263157893), (0.40871710526315785, 0.0)).lineTo(0.40871710526315785, 0.546875).lineTo(0.2763157894736842, 0.546875).threePointArc((0.20723684210526314, 0.4777960526315789), (0.1381578947368421, 0.546875)).lineTo(0.0, 0.546875).lineTo(0.0, 0.0).close()
loop2=wp_sketch1.moveTo(0.20723684210526314, 0.0).circle(0.06907894736842105)
solid1=wp_sketch1.add(loop1).add(loop2).extrude(0.1328125)
solid=solid.union(solid1)