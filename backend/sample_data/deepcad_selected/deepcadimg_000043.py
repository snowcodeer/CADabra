import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.34375, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.34342105263157896, 0.0).circle(0.34342105263157896)
loop1=wp_sketch0.moveTo(0.34342105263157896, 0.0).circle(0.19317434210526316)
loop2=wp_sketch0.moveTo(0.20748355263157894, -0.22894736842105262).circle(0.028618421052631578)
loop3=wp_sketch0.moveTo(0.20748355263157894, 0.22894736842105262).circle(0.028618421052631578)
loop4=wp_sketch0.moveTo(0.6009868421052631, 0.0).circle(0.028618421052631578)
solid0=wp_sketch0.add(loop0).add(loop1).add(loop2).add(loop3).add(loop4).extrude(0.1015625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.1953125, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop5=wp_sketch1.moveTo(0.19736842105263158, 0.0).circle(0.19736842105263158)
loop6=wp_sketch1.moveTo(0.19736842105263158, 0.0).circle(0.16036184210526314)
solid1=wp_sketch1.add(loop5).add(loop6).extrude(0.75)
solid=solid.union(solid1)