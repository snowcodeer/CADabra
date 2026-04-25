import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5, -0.75, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.9947368421052631, 0.0).lineTo(0.9947368421052631, 1.5).lineTo(0.0, 1.5).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.7894736842105263, 0.2210526315789474).lineTo(0.7894736842105263, 1.2789473684210526).lineTo(0.2210526315789474, 1.2789473684210526).lineTo(0.2210526315789474, 0.2210526315789474).close()
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.21875)
solid=solid0