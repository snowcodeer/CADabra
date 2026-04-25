import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.6015625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.6, 0.0).lineTo(0.75, 0.15).lineTo(0.75, 0.45).lineTo(0.6, 0.6).lineTo(0.0, 0.6).threePointArc((0.3, 0.3), (0.0, 0.0)).close()
solid0=wp_sketch0.add(loop0).extrude(0.6015625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.6015625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.0, 0.15197368421052632).threePointArc((-0.14880756578947368, 0.30078125), (0.0, 0.4495888157894737)).lineTo(0.0, 0.6015625).threePointArc((-0.30078125, 0.30078125), (0.0, 0.0)).close()
solid1=wp_sketch1.add(loop1).extrude(0.6015625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.0, -0.6015625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.0, 0.0).threePointArc((0.30078125, 0.30078125), (0.0, 0.6015625)).lineTo(0.0, 0.4495888157894737).threePointArc((0.14880756578947368, 0.30078125), (0.0, 0.15197368421052632)).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.6015625)
solid=solid.union(solid2)