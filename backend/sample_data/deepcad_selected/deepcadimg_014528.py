import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.1796875, 0.0, 0.265625), cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17), cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)))
loop0=wp_sketch0.moveTo(0.17615131578947368, 0.0).lineTo(0.7535361842105263, 0.0).lineTo(0.9296874999999999, 0.0).lineTo(0.9296874999999999, 0.21529605263157892).lineTo(0.7535361842105263, 0.21529605263157892).lineTo(0.17615131578947368, 0.21529605263157892).lineTo(0.0, 0.21529605263157892).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.4609375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(0.0546875, -0.234375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(0.22894736842105262, 0.0).circle(0.22894736842105262)
loop2=wp_sketch1.moveTo(0.22894736842105262, 0.0).circle(0.13355263157894737)
solid1=wp_sketch1.add(loop1).add(loop2).extrude(0.6953125)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(0.1015625, -0.234375, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop3=wp_sketch2.moveTo(0.18157894736842106, 0.0).circle(0.18157894736842106)
solid2=wp_sketch2.add(loop3).extrude(0.6015625, both=True)
solid=solid.cut(solid2)