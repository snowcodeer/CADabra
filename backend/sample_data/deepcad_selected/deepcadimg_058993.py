import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.203125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.5, 0.0).lineTo(1.5, 0.4105263157894737).lineTo(0.0, 0.4105263157894737).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.0234375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.203125, 0.0234375), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(1.5, 0.0).lineTo(1.5, 0.4105263157894737).lineTo(0.0, 0.4105263157894737).lineTo(0.0, 0.0).close()
loop2=wp_sketch1.moveTo(1.4842105263157894, 0.015789473684210527).lineTo(1.4842105263157894, 0.12631578947368421).lineTo(0.015789473684210527, 0.26842105263157895).lineTo(0.015789473684210527, 0.015789473684210527).close()
loop3=wp_sketch1.moveTo(1.4842105263157894, 0.14210526315789473).lineTo(1.4842105263157894, 0.37894736842105264).lineTo(0.015789473684210527, 0.37894736842105264).lineTo(0.015789473684210527, 0.28421052631578947).close()
solid1=wp_sketch1.add(loop1).add(loop2).add(loop3).extrude(0.0703125)
solid=solid.union(solid1)