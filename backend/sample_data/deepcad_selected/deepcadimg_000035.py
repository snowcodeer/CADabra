import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.375, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.37894736842105264, 0.0).circle(0.37894736842105264)
loop1=wp_sketch0.moveTo(0.37894736842105264, 0.0).circle(0.3157894736842105)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.5)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.5, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(0.5052631578947369, 0.0).circle(0.5052631578947369)
loop3=wp_sketch1.moveTo(0.5052631578947369, 0.0).circle(0.37894736842105264)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.625, -0.4140625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop4=wp_sketch2.moveTo(0.0, 0.0).threePointArc((0.625, -0.33406945996924503), (1.25, 0.0)).lineTo(1.25, 0.8289473684210525).threePointArc((0.625, 1.1630168283902975), (0.0, 0.8289473684210525)).lineTo(0.0, 0.0).close()
loop5=wp_sketch2.moveTo(0.631578947368421, 0.42105263157894735).circle(0.5)
solid2=wp_sketch2.add(loop4).add(loop5).extrude(0.25)
solid=solid.union(solid2)