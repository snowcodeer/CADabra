import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.234375, 0.0390625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.01875, -0.140625).lineTo(0.11249999999999999, -0.23906249999999998).lineTo(0.253125, -0.26718749999999997).lineTo(0.3796875, -0.20625).lineTo(0.4453125, -0.08437499999999999).lineTo(0.4265625, 0.056249999999999994).lineTo(0.3328125, 0.159375).lineTo(0.19218749999999998, 0.1828125).lineTo(0.065625, 0.121875).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.25)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.1328125, -0.0234375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.04613486842105263, -0.07598684210526316).lineTo(0.13297697368421055, -0.10312500000000001).lineTo(0.21439144736842106, -0.0705592105263158).lineTo(0.2578125, 0.008141447368421054).lineTo(0.23881578947368423, 0.09498355263157895).lineTo(0.16825657894736842, 0.15197368421052632).lineTo(0.07870065789473685, 0.14925986842105265).lineTo(0.013569078947368422, 0.08955592105263159).lineTo(0.0, 0.0).close()
solid1=wp_sketch1.add(loop1).extrude(0.5)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.078125, -0.0234375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch2.moveTo(0.041118421052631575, -0.04769736842105263).lineTo(0.10361842105263157, -0.05263157894736842).lineTo(0.1513157894736842, -0.013157894736842105).lineTo(0.15625, 0.049342105263157895).lineTo(0.11513157894736842, 0.09703947368421052).lineTo(0.05263157894736842, 0.10361842105263157).lineTo(0.004934210526315789, 0.0625).lineTo(0.0, 0.0).close()
solid2=wp_sketch2.add(loop2).extrude(0.75)
solid=solid.union(solid2)