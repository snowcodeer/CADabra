import cadquery as cq
w0=cq.Workplane('YZ',origin=(-1,0,0))
r=w0.workplane(offset=-13/2).moveTo(24,3).cylinder(13,12).union(w0.sketch().segment((-50,-8),(-26,-19)).segment((-26,-26)).segment((30,-26)).segment((30,2)).segment((50,2)).segment((50,4)).segment((30,4)).segment((30,26)).segment((-26,26)).segment((-26,21)).segment((-35,25)).close().assemble().finalize().extrude(15))