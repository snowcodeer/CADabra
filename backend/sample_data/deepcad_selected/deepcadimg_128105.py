import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.1015625, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.09868421052631579, 0.0).circle(0.09868421052631579)
loop1=wp_sketch0.moveTo(0.09868421052631579, 0.0).circle(0.047286184210526314)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.1953125)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.078125, 0.0, -0.0625), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop2=wp_sketch1.moveTo(0.546875, 0.0).threePointArc((0.5150011982857355, 0.06332236842105263), (0.546875, 0.12664473684210525)).lineTo(0.0, 0.12664473684210525).threePointArc((0.0230963597353962, 0.06332236842105263), (0.0, 0.0)).close()
solid1=wp_sketch1.add(loop2).extrude(0.1953125)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.59375, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop3=wp_sketch2.moveTo(0.07894736842105263, 0.0).circle(0.07894736842105263)
loop4=wp_sketch2.moveTo(0.07894736842105263, 0.0).circle(0.04769736842105263)
solid2=wp_sketch2.add(loop3).add(loop4).extrude(0.1953125)
solid=solid.union(solid2)