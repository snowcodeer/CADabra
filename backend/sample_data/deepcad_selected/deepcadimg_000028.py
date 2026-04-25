import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.5, -0.6015625, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.0, 0.0).threePointArc((0.04583682527976966, -0.11065988524654614), (0.1564967105263158, -0.1564967105263158)).lineTo(0.8536184210526316, -0.1564967105263158).threePointArc((0.9571648194570725, -0.1077133825207835), (0.995888157894737, 0.0)).lineTo(0.995888157894737, 1.1950657894736842).threePointArc((0.9571648194570725, 1.3027791719944677), (0.8536184210526316, 1.3515625)).lineTo(0.1564967105263158, 1.3515625).threePointArc((0.04583682527976966, 1.3057256747202304), (0.0, 1.1950657894736842)).lineTo(0.0, 0.0).close()
loop1=wp_sketch0.moveTo(0.4979440789473685, 0.5975328947368421).circle(0.2560855263157895)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.15625)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.34375, 0.0, 0.15625), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(0.34736842105263155, 0.0).circle(0.34736842105263155)
loop3=wp_sketch1.moveTo(0.34736842105263155, 0.0).circle(0.24605263157894736)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.140625)
solid=solid.union(solid1)