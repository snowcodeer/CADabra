import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.375, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.18947368421052632, -0.3236842105263158).lineTo(0.5605263157894737, -0.3236842105263158).lineTo(0.75, 0.0).lineTo(0.5605263157894737, 0.3236842105263158).lineTo(0.18947368421052632, 0.3236842105263158).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.20526315789473684, -0.3).lineTo(0.5447368421052632, -0.3).lineTo(0.718421052631579, 0.0).lineTo(0.5447368421052632, 0.3).lineTo(0.20526315789473684, 0.3).lineTo(0.031578947368421054, 0.0).close()
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.75)
solid=solid0