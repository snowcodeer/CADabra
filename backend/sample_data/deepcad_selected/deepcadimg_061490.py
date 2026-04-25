import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.46875, 0.25, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.5, 0.0).lineTo(0.5, 0.17368421052631577).lineTo(0.4052631578947368, 0.17368421052631577).lineTo(0.4052631578947368, 0.3263157894736842).lineTo(0.5, 0.3263157894736842).lineTo(0.5, 0.5).lineTo(0.0, 0.5).lineTo(0.0, 0.3263157894736842).lineTo(0.09473684210526316, 0.3263157894736842).lineTo(0.09473684210526316, 0.17368421052631577).lineTo(0.0, 0.17368421052631577).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.25263157894736843, 0.25263157894736843).circle(0.05789473684210526)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.46875, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(0.5, 0.0).lineTo(0.5, 0.25263157894736843).lineTo(0.0, 0.25263157894736843).lineTo(0.0, 0.0).close()
loop3=wp_sketch1.moveTo(0.25263157894736843, 0.12631578947368421).circle(0.06315789473684211)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.125)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.46875, 0.4296875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop4=wp_sketch2.moveTo(0.09375, 0.0).lineTo(0.09375, 0.1484375).lineTo(0.0, 0.1484375).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop4).extrude(0.25)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(-0.0625, 0.4296875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop5=wp_sketch3.moveTo(0.09375, 0.0).lineTo(0.09375, 0.1484375).lineTo(0.0, 0.1484375).lineTo(0.0, 0.0).close()
solid3=wp_sketch3.add(loop5).extrude(0.25)
solid=solid.union(solid3)