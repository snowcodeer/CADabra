import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.1875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.5, 0.0).lineTo(1.5, 0.37894736842105264).lineTo(0.0, 0.37894736842105264).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.2890625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.0859375, 0.0, 0.2890625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.08289473684210526, 0.0).circle(0.08289473684210526)
solid1=wp_sketch1.add(loop1).extrude(0.2890625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.2578125, 0.0, 0.2890625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.11842105263157894, 0.0).circle(0.11595394736842105)
solid2=wp_sketch2.add(loop2).extrude(0.2890625)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(-0.4765625, 0.0, 0.2890625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch3.moveTo(0.07894736842105263, 0.0).circle(0.07894736842105263)
solid3=wp_sketch3.add(loop3).extrude(0.2890625)
solid=solid.union(solid3)