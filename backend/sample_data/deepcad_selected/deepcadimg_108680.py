import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.3125, 0.0, -0.2734375), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.625, 0.0).lineTo(0.625, 0.625).lineTo(0.0, 0.625).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.3157894736842105, 0.30921052631578944).circle(0.1907894736842105)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.5)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.1875, 0.0, 0.0390625), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop2=wp_sketch1.moveTo(0.18947368421052632, 0.0).circle(0.18947368421052632)
loop3=wp_sketch1.moveTo(0.18947368421052632, 0.0).circle(0.15)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.75)
solid=solid.union(solid1)