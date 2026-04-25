import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.4765625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.23832236842105264, 0.0).lineTo(0.23832236842105264, 0.359375).lineTo(0.0, 0.359375).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.3203125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.75, -0.1171875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.23832236842105264, 0.0).lineTo(0.23832236842105264, 0.359375).lineTo(0.0, 0.359375).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.4765625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.515625, -0.4765625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.9875, 0.0).lineTo(0.9875, 0.47500000000000003).lineTo(1.1875, 0.47500000000000003).lineTo(1.1875, 0.7125).lineTo(0.9875, 0.7125).lineTo(0.0, 0.7125).lineTo(0.0, 0.35000000000000003).lineTo(0.0, 0.0).close()
loop3=wp_sketch2.moveTo(0.47500000000000003, 0.35000000000000003).circle(0.1375)
solid2=wp_sketch2.add(loop2).add(loop3).extrude(0.234375)
solid=solid.union(solid2)