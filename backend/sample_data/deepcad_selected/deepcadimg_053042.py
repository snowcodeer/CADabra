import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.75, 0.0).lineTo(0.75, 0.25263157894736843).lineTo(0.0, 0.25263157894736843).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.18947368421052632, 0.14210526315789473).circle(0.031578947368421054)
loop2=wp_sketch0.moveTo(0.37894736842105264, 0.1736842105263158).circle(0.031578947368421054)
loop3=wp_sketch0.moveTo(0.5526315789473685, 0.13421052631578947).circle(0.031578947368421054)
solid0=wp_sketch0.add(loop0).add(loop1).add(loop2).add(loop3).extrude(0.0234375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0, 0.0, 0.0), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop4=wp_sketch1.moveTo(0.75, 0.0).lineTo(0.75, 0.25263157894736843).lineTo(0.0, 0.25263157894736843).lineTo(0.0, 0.0).close()
loop5=wp_sketch1.moveTo(0.18947368421052632, 0.14210526315789473).circle(0.031578947368421054)
loop6=wp_sketch1.moveTo(0.37894736842105264, 0.1736842105263158).circle(0.031578947368421054)
loop7=wp_sketch1.moveTo(0.5526315789473685, 0.13421052631578947).circle(0.031578947368421054)
solid1=wp_sketch1.add(loop4).add(loop5).add(loop6).add(loop7).extrude(-0.25)
solid=solid.cut(solid1)