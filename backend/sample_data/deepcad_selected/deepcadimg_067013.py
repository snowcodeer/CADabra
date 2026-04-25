import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.4609375, 0.1484375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.17615131578947368, -0.5480263157894736).lineTo(0.7535361842105263, -0.5480263157894736).lineTo(0.9296874999999999, 0.0).lineTo(0.4697368421052631, 0.342516447368421).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.1957236842105263, -0.5186677631578946).lineTo(0.7339638157894737, -0.5186677631578946).lineTo(0.8905427631578947, -0.009786184210526315).lineTo(0.4697368421052631, 0.3033717105263158).lineTo(0.03914473684210526, -0.009786184210526315).close()
solid0=wp_sketch0.add(loop0).add(loop1).extrude(-0.1484375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.4296875, 0.140625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(0.16282894736842107, -0.506578947368421).lineTo(0.696546052631579, -0.506578947368421).lineTo(0.859375, 0.0).lineTo(0.4342105263157895, 0.31661184210526316).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop2).extrude(-0.1484375)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.328125, -0.0625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch2.moveTo(0.027960526315789474, -0.07689144736842106).lineTo(0.2935855263157895, -0.27261513157894735).lineTo(0.3704769736842105, -0.27261513157894735).lineTo(0.6361019736842105, -0.07689144736842106).lineTo(0.6640625, 0.0).lineTo(0.5662006578947368, 0.3145559210526316).lineTo(0.49629934210526316, 0.36348684210526316).lineTo(0.16776315789473684, 0.36348684210526316).lineTo(0.09786184210526316, 0.3145559210526316).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop3).extrude(0.75)
solid=solid.union(solid2)