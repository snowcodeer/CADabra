import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5546875, -0.421875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(1.2890625, 0.0).lineTo(1.2890625, 0.8005756578947368).lineTo(0.0, 0.8005756578947368).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(1.2619243421052633, 0.027138157894736843).lineTo(1.2619243421052633, 0.7734375).lineTo(0.027138157894736843, 0.7734375).lineTo(0.027138157894736843, 0.027138157894736843).close()
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.09375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.578125, -0.4375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(1.328125, 0.0).lineTo(1.328125, 0.852796052631579).lineTo(0.0, 0.852796052631579).lineTo(0.0, 0.0).close()
loop3=wp_sketch1.moveTo(1.3141447368421053, 0.013980263157894737).lineTo(1.3141447368421053, 0.8248355263157895).lineTo(0.013980263157894737, 0.8248355263157895).lineTo(0.013980263157894737, 0.013980263157894737).close()
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.125)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.5234375, -0.390625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop4=wp_sketch2.moveTo(1.2265625, 0.0).lineTo(1.2265625, 0.7488486842105263).lineTo(0.0, 0.7488486842105263).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop4).extrude(0.03125)
solid=solid.union(solid2)