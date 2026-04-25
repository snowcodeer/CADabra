import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5, 0.0, -0.5), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(1.0, 0.0).lineTo(1.0, 1.0).lineTo(0.0, 1.0).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.5052631578947369, 0.5052631578947369).circle(0.3157894736842105)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.75, 0.0, -0.75), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop2=wp_sketch1.moveTo(1.5, 0.0).lineTo(1.5, 1.5).lineTo(0.031578947368421054, 1.5).lineTo(0.031578947368421054, 1.4842105263157894).lineTo(0.0, 1.4842105263157894).lineTo(0.0, 0.0).close()
loop3=wp_sketch1.moveTo(0.7578947368421053, 0.7578947368421053).circle(0.3157894736842105)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(-0.5)
solid=solid.union(solid1)