import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.09375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.5, 0.0).lineTo(1.5, 0.18947368421052632).lineTo(0.0, 0.18947368421052632).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(1.436842105263158, 0.07894736842105263).lineTo(1.436842105263158, 0.1105263157894737).lineTo(0.06315789473684211, 0.1105263157894737).lineTo(0.06315789473684211, 0.07894736842105263).close()
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.0234375)
solid=solid0