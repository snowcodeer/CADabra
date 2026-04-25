import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5390625, 0.0, -0.2109375), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.0, 0.10855263157894737).threePointArc((-0.10629111842105264, 0.21484375), (0.0, 0.32113486842105265)).lineTo(0.0, 0.4296875).threePointArc((-0.21484375, 0.21484375), (0.0, 0.0)).close()
solid0=wp_sketch0.add(loop0).extrude(0.1640625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.5390625, 0.0, -0.2109375), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.5390625, 0.0).lineTo(0.5390625, 0.14185855263157895).threePointArc((0.46529605263157897, 0.215625), (0.5390625, 0.28939144736842104)).lineTo(0.5390625, 0.43125).lineTo(0.0, 0.43125).lineTo(0.0, 0.32343750000000004).threePointArc((0.1078125, 0.215625), (0.0, 0.1078125)).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.1640625)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.0, -0.2109375), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop2=wp_sketch2.moveTo(0.0, 0.0).threePointArc((0.21484375, 0.21484375), (0.0, 0.4296875)).lineTo(0.0, 0.2894736842105263).threePointArc((0.07462993421052631, 0.21484375), (0.0, 0.1402138157894737)).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.1640625)
solid=solid.union(solid2)