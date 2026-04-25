import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.1015625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.10263157894736842, 0.0).lineTo(0.10263157894736842, -0.10263157894736842).lineTo(0.6473684210526316, -0.10263157894736842).lineTo(0.6473684210526316, 0.0).lineTo(0.75, 0.0).lineTo(0.75, 0.3394736842105263).lineTo(0.6473684210526316, 0.3394736842105263).lineTo(0.6473684210526316, 0.37894736842105264).lineTo(0.10263157894736842, 0.37894736842105264).lineTo(0.10263157894736842, 0.3394736842105263).lineTo(0.0, 0.3394736842105263).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.0390625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.1015625, 0.0, 0.0390625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.546875, 0.0).lineTo(0.546875, 0.04029605263157895).lineTo(0.0, 0.04029605263157895).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.3203125)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.3984375, 0.0390625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.1015625, 0.0).lineTo(0.1015625, 0.03848684210526316).lineTo(0.0, 0.03848684210526316).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.3203125)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(0.6484375, 0.3984375, 0.0390625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch3.moveTo(0.1015625, 0.0).lineTo(0.1015625, 0.03848684210526316).lineTo(0.0, 0.03848684210526316).lineTo(0.0, 0.0).close()
solid3=wp_sketch3.add(loop3).extrude(0.3203125)
solid=solid.union(solid3)