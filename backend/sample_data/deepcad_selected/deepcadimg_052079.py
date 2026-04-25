import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.7578947368421053, 0.0).lineTo(0.9789473684210527, 0.0).lineTo(1.5, 0.0).lineTo(1.5, 0.7578947368421053).lineTo(0.0, 0.7578947368421053).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(1.0105263157894737, 0.18947368421052632).circle(0.06315789473684211)
loop2=wp_sketch0.moveTo(1.0105263157894737, 0.5684210526315789).circle(0.06315789473684211)
loop3=wp_sketch0.moveTo(1.3105263157894738, 0.18947368421052632).circle(0.06315789473684211)
loop4=wp_sketch0.moveTo(1.3105263157894738, 0.5684210526315789).circle(0.06315789473684211)
solid0=wp_sketch0.add(loop0).add(loop1).add(loop2).add(loop3).add(loop4).extrude(0.5234375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.5234375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop5=wp_sketch1.moveTo(0.2265625, 0.0).lineTo(0.2265625, 0.1502467105263158).lineTo(0.0, 0.1502467105263158).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop5).extrude(0.5234375)
solid=solid.union(solid1)