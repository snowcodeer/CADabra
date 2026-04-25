import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.34375, 0.0, -0.1953125), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.6633223684210526, 0.0).lineTo(0.6633223684210526, 0.18824013157894737).lineTo(0.47508223684210527, 0.18824013157894737).lineTo(0.47508223684210527, 0.6633223684210526).lineTo(0.6633223684210526, 0.6633223684210526).lineTo(0.6633223684210526, 0.8515625).lineTo(0.0, 0.8515625).lineTo(0.0, 0.6633223684210526).lineTo(0.18824013157894737, 0.6633223684210526).lineTo(0.18824013157894737, 0.18824013157894737).lineTo(0.0, 0.18824013157894737).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.5625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.125, -0.390625, 0.328125), cq.Vector(3.749399456654644e-33, 1.0, -6.123233995736766e-17), cq.Vector(1.0, 0.0, 6.123233995736766e-17)))
loop1=wp_sketch1.moveTo(0.11052631578947367, 0.0).circle(0.11052631578947367)
solid1=wp_sketch1.add(loop1).extrude(-0.9453125)
solid=solid.cut(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.34375, -0.0859375, 0.65625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.6640625, 0.0).lineTo(0.6640625, 0.09087171052631579).lineTo(0.0, 0.09087171052631579).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.09375)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(-0.34375, -0.5390625, 0.65625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch3.moveTo(0.6640625, 0.0).lineTo(0.6640625, 0.09786184210526316).lineTo(0.0, 0.09786184210526316).lineTo(0.0, 0.0).close()
solid3=wp_sketch3.add(loop3).extrude(0.09375)
solid=solid.union(solid3)