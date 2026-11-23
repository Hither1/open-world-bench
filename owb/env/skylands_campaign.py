"""Full-scale Skylands campaigns with regional logistics and complete travel.

Shares tested local mechanics with the compact testbed, but authors a separate
world: regional roads/caverns/islands, real ascent/descent connections, several
return moorings, separate regional reserves, and navigable cargo waterways.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import heapq
import hashlib
import json
import numpy as np
from scipy.ndimage import binary_dilation
from .skylands_reference import SkylandsReferenceEnv, Site, A, B_TASK, plain, COMMANDS as REFERENCE_COMMANDS
from ..sim.riding import COMMANDS as RIDING_COMMANDS
from ..sim.orchard import Orchard
from ..world import organic as O
from ..world.fauna import populate
from ..core import blocks as B, noise as N
from ..world.voxel import VoxelWorld
from ..world.worldgen import WorldGenResult, Landmark
from ..world.skylands import _island, _tree

A_PLACES=[(90,90),(170,100),(300,130),(400,170),(480,240),(80,500),(500,400),
          (430,400),(80,410),(120,430),(150,430),(430,400),(80,500),(460,520),(460,520)]
B_PLACES=[(90,90),(170,100),(300,130),(400,170),(480,240),(300,300),(150,330),
          (400,100),(90,120),(480,440),(70,430),(100,500),(150,500),(430,470),
          (470,470),(300,360),(480,360),(450,360),(450,450),(200,540),
          (250,540),(290,540),(350,540),(350,540)]


FAN_SWEET=(0.5,1.5)          # seconds after the previous strike for a clean spin-up
TRAIL_ORCHARD_SPACING=40.0   # metres between planted trail-side apple trees
FAN_GOOD,FAN_LATE,FAN_JAM=45.,12.,-10.


class SkylandsCampaignEnv(SkylandsReferenceEnv):
    levels=(8,48,112)
    presentation_kind='skylands_campaign'
    #: Organic roads/pads/canals, creatures, riding, orchards and fan striking.
    #: Subclasses with published traces (tempest, restoration) switch it off.
    organic_world=True
    world_features=frozenset({'riding','orchard'})
    # Riding commands are appended last: traces store integer command indices.
    commands=REFERENCE_COMMANDS+RIDING_COMMANDS

    def __init__(self,task='a',seed=7,*,render_enabled=False):
        self.geography_scale=1.5 if task=='a' else 2.0
        super().__init__(task,seed,render_enabled=render_enabled)
        self.cfg.world_cfg=replace(self.cfg.world_cfg,size_x=int(640*self.geography_scale),
                                  size_y=144,size_z=int(704*self.geography_scale))
        self.cfg.max_steps=80000
        self.cfg.render_cfg.max_dist=300
        self.vertical_xy=self.point(47,140)
        self.descent_xy=self.point(40,120)

    def point(self,x,z): return (int(round(x*self.geography_scale)),int(round(z*self.geography_scale)))

    def site_specs(self):
        rows=A if self.task=='a' else B_TASK; places=A_PLACES if self.task=='a' else B_PLACES
        return [(title,kind,layer,*self.point(x,z)) for (title,kind,layer,_,_),(x,z) in zip(rows,places)]

    def boat_length(self,m):
        base={'A02':70,'A03':80,'A06':10,'A11':260,'B02':70,'B03':80,'B13':220,'B16':150,'B23':160}
        return int(base.get(m.id,80)*self.geography_scale)

    def shuttle_span(self,m): return 26 if m.id in ('B06','B07') else 34

    def _generate_world(self):
        self._protect_boxes=[]; self._canal_paths=[]
        sx,sy,sz=self.cfg.world_cfg.size_x,self.cfg.world_cfg.size_y,self.cfg.world_cfg.size_z
        w=VoxelWorld(sx,sy,sz,sea_level=0); self._road_mask=np.zeros((sx,sz),bool)
        # Trails per layer (the shared _road_mask mixes cavern, surface and sky).
        self._layer_roads={k:np.zeros((sx,sz),bool) for k in range(3)}; self._surface_roads=self._layer_roads[1]
        xx=np.arange(sx)[:,None]; zz=np.arange(sz)[None,:]
        h=(47+5*np.sin(xx/77)*np.cos(zz/61)+3*np.sin((xx+zz)/121)).astype(np.int16)
        w.fill_box((0,0,0),(sx-1,7,sz-1),B.BASALT)
        # The surface is an irregular stone roof over connected deep caverns.
        for x in range(sx):
            for offset in range(4): w.vox[x,h[x]-offset,np.arange(sz)]=B.STONE
            w.vox[x,h[x],np.arange(sz)]=B.GRASS
        specs=self.site_specs()
        hubs={0:self.point(60,180),1:self.point(60,180),2:self.point(60,180)}
        for _,_,layer,x,z in specs:
            y=self.levels[layer]
            if layer==2: _island(w,x,z,48,42,y-1,27,(x+z)*.013)
            self._pad(w,(x,y,z),radius=12)
        # A connected regional trail, rather than the compact course's tile grid.
        # Surface tracks cut through hills; sky tracks are physical stone bridges.
        for layer in range(3):
            positions=[(x,z) for _,_,l,x,z in specs if l==layer]
            positions += [self.vertical_xy,self.descent_xy]
            if layer==0: positions += [self.point(80,100),self.point(120,140),self.point(160,100)]
            if layer==1: positions += [self.point(70,110),self.point(90,130),self.point(110,110)]
            if self.organic_world:
                self._trail_network(w,layer,hubs[layer],positions)
            else:
                for x,z in positions:
                    hx,hz=hubs[layer]
                    bend=(hx,int((hz+z)/2))
                    self._road(w,layer,[(x,z),(x,hz),bend,(hx,hz)])
        # Dense, stable botanical stands and irregular rock/column silhouettes.
        rng=np.random.default_rng(self._seed+71777)
        for _ in range(int(170*self.geography_scale)):
            x=int(rng.integers(20,sx-20)); z=int(rng.integers(20,sz-20))
            if self._road_mask[max(0,x-5):x+6,max(0,z-5):z+6].any(): continue
            y=int(h[x,z])+1
            if any(l==1 and abs(x-px)<25 and abs(z-pz)<25 for _,_,l,px,pz in specs): continue
            _tree(w,x,y,z,int(rng.integers(5,10)))
        for i in range(int(65*self.geography_scale)):
            x=int(rng.integers(20,sx-20)); z=int(rng.integers(20,sz-20))
            if self._road_mask[x-7:x+8,z-7:z+8].any(): continue
            if any(abs(x-px)<24 and abs(z-pz)<24 for _,_,_,px,pz in specs): continue
            width=int(rng.integers(2,5)); height=int(rng.integers(14,31))
            for dx in range(-width,width+1):
                for dz in range(-width,width+1):
                    if dx*dx+dz*dz<(width+.3)**2:
                        top=8+height-abs(dx)-abs(dz)
                        w.fill_box((x+dx,8,z+dz),(x+dx,top,z+dz),B.BASALT)
            w.set_block(x-width-1,8,z,B.CRYSTAL)
        for _,_,layer,x,z in specs:
            if layer==2:
                for dx,dz in ((-20,-18),(22,15),(-24,19)):
                    if B.SOLID[w.get(x+dx,111,z+dz)]: _tree(w,x+dx,112,z+dz,5+(x+z)%3)
        self._author_vertical_connections(w)
        w.temperature[:]=.8; w.moisture[:]=.5; w.biome[:]=0
        w.meta.update(theme='skylands',layered=True,layer_heights=(7,47,111),
            bridge=(10,111,10,6),beacon=(12,112,12),parts=[],scenery=[],
            crystal_lights=[tuple(p+.5) for p in np.argwhere(w.vox==B.CRYSTAL)])
        w.set_region_names({0:'Aster Reach campaign'})
        w.recompute_heightmaps()
        x,z=self.point(90,90); spawn=(x-2.5,48.01,z+.5)
        return WorldGenResult(w,spawn,landmarks=[Landmark('camp','Aster Reach landing',spawn,5)])

    def _pad(self,w,pos,radius=10):
        if self.organic_world: return self._organic_pad(w,pos,radius)
        x,y,z=map(int,pos)
        w.fill_box((x-radius,y-1,z-radius),(x+radius,y-1,z+radius),B.MOSS_STONE if y<30 else B.GRASS)
        w.fill_box((x-radius,y,z-radius),(x+radius,y+9,z+radius),B.AIR)

    # ------------------------------------------------------------------
    # Organic geometry (organic_world): curved trails with rounded corners and
    # graded banks, irregular pads, rounded canals held below their banks.
    def _protected(self,layer):
        shape=(self.cfg.world_cfg.size_x,self.cfg.world_cfg.size_z); m=np.zeros(shape,bool)
        dx,dz=self.descent_xy
        if layer>=1: m[dx-4:dx+5,dz-4:dz+5]=True
        if layer==2: m[dx+14:dx+25,dz-5:dz+6]=True
        for lo,hi in getattr(self,'_protect_boxes',[]):
            if lo[1]<=self.levels[layer]-1<=hi[1]: m[lo[0]:hi[0]+1,lo[2]:hi[2]+1]=True
        return m

    def _surface_grade(self,w,mask,level,rim=0.0):
        O.grade_band(w,mask,level,('skylands-campaign',self._seed,self.task),ylo=30,yhi=82,
                     protect=self._protected(1)|(w.vox[:,level,:]==B.WATER),rim=rim)

    @staticmethod
    def _fill_under(w,xs,zs,y):
        """Earth under a surface trail or pad down to the natural ground."""
        band=B.SOLID[w.vox[xs,30:y-2,zs]]
        has=band.any(axis=1); top=30+band.shape[1]-1-np.argmax(band[:,::-1],axis=1)
        for x,z,h in zip(xs[has],zs[has],top[has]):
            w.vox[x,h+1:y-1,z]=B.DIRT

    def _organic_road(self,w,layer,points,meander=1.2,radius=6.0):
        """Trail through the given control points at any angle (not split into
        axis-aligned legs), with rounded bends, a lateral meander and a varying
        width. Surface trails are gravel with dirt verges."""
        y=self.levels[layer]
        identity=('skylands-road',self._seed,self.task,layer,tuple(map(tuple,np.asarray(points,int).tolist())))
        road=O.winding_road(np.asarray(points,float)+.5,identity,radius=radius,meander=meander)
        mask=road.stamp((w.size_x,w.size_z))&~self._protected(layer)
        mask&=w.vox[:,y-1,:]!=B.WATER
        xs,zs=np.nonzero(mask)
        if layer==1:
            # Packed dirt along the verges and in patches; gravel down the middle.
            inner=O.ndimage.binary_erosion(mask,iterations=1)[xs,zs]
            patch=N.hash_unit(xs//3,zs//3,O.stable_seed('trail-patch',self._seed)%100003)<.18
            w.vox[xs,y-1,zs]=np.where(inner&~patch,B.GRAVEL,B.DIRT)
        else:
            w.vox[xs,y-1,zs]=B.BRICK
        for dy in range(3 if layer!=1 else 12):
            col=w.vox[xs,y+dy,zs]
            # The authored three cells of headroom are always clear; above that
            # only natural ground is cut, never built structures.
            clear=(col!=B.WATER) if dy<3 else O.CUTTABLE[col]
            w.vox[xs,y+dy,zs]=np.where(clear,B.AIR,col)
        if layer==1:
            self._fill_under(w,xs,zs,y)
            self._surface_grade(w,mask,y-1)
        self._road_mask|=mask
        self._layer_roads[layer]|=mask

    def _organic_pad(self,w,pos,radius):
        x,y,z=map(int,pos)
        identity=('skylands-pad',self._seed,self.task,x,y,z)
        shape=(w.size_x,w.size_z)
        lo0,lo1=max(0,x-2*radius-3),max(0,z-2*radius-3)
        hi0,hi1=min(shape[0],x+2*radius+4),min(shape[1],z+2*radius+4)
        local=O.blob((hi0-lo0,hi1-lo1),(x-lo0+.5,z-lo1+.5),(radius+.6,radius+.6),identity,exponent=2.3,amplitude=.16)
        mask=np.zeros(shape,bool); mask[lo0:hi0,lo1:hi1]=local
        mask|=O.rounded_box(shape,(x-max(1,radius//2),z-max(1,radius//2)),(x+max(1,radius//2)+1,z+max(1,radius//2)+1),.5)
        mask&=~self._protected(self.levels.index(y) if y in self.levels else 1)
        xs,zs=np.nonzero(mask)
        w.vox[xs,y-1,zs]=B.MOSS_STONE if y<30 else B.GRASS
        for dy in range(10): w.vox[xs,y+dy,zs]=B.AIR
        if y==self.levels[1]:
            self._fill_under(w,xs,zs,y)
            self._surface_grade(w,mask,y-1)

    def _organic_water(self,points,width):
        w=self.world; y=int(points[0][1])
        legs=[(float(p[0]),float(p[2])) for p in points]
        identity=('skylands-canal',self._seed,self.task,tuple(np.round(legs,1).tolist()),width)
        if y==self.levels[1]: legs=self._meander_legs(legs,identity)
        path,_=O.rounded_corners(legs,radius=width+3)
        # Rounded bends around the authored grid legs the raft steps along; the
        # footprint otherwise matches the authored canal, so docks keep their
        # clearance. Banks get irregular graded slopes below.
        channel=O.Road(path,float(width)).stamp((w.size_x,w.size_z))
        core=np.zeros((w.size_x,w.size_z),bool)
        for (ax,az),(bx,bz) in zip(legs,legs[1:]):
            x0,x1=sorted((int(ax),int(bx))); z0,z1=sorted((int(az),int(bz)))
            if az==bz: core[x0-1:x1+2,z0-width:z1+width+1]=True
            else: core[x0-width:x1+width+1,z0-1:z1+2]=True
        channel&=binary_dilation(core,iterations=1)
        mask=channel|core
        shore=self._natural_shore(w,mask,legs,identity) if y==self.levels[1] else np.zeros_like(mask)
        xs,zs=np.nonzero(mask)
        w.vox[xs,y-3,zs]=B.WATER; w.vox[xs,y-2,zs]=B.WATER; w.vox[xs,y-1,zs]=B.WATER
        for dy in range(5): w.vox[xs,y+dy,zs]=B.AIR
        w.vox[xs,y-4,zs]=np.where(B.SOLID[w.vox[xs,y-4,zs]],w.vox[xs,y-4,zs],B.GRAVEL)
        # Shallow natural margin: one cell of water over a sand bed, sealed below.
        sx_,sz_=np.nonzero(shore)
        if len(sx_):
            w.vox[sx_,y-1,sz_]=B.WATER; w.vox[sx_,y-2,sz_]=B.SAND
            for dy in (3,4):
                col=w.vox[sx_,y-dy,sz_]; w.vox[sx_,y-dy,sz_]=np.where(B.SOLID[col],col,B.DIRT)
            for dy in range(5):
                col=w.vox[sx_,y+dy,sz_]; w.vox[sx_,y+dy,sz_]=np.where(O.CUTTABLE[col],B.AIR,col)
        water=mask|shore
        if y==self.levels[1]: water|=self._refill_base_channel(w,water,points,width,y)
        # Seal the channel: a one-cell wall along the sides down to the bed,
        # wherever the canal is cut through hollow ground (cavern air below
        # the surface roof), so the water is always held.
        ring=binary_dilation(water,iterations=1)&~water
        rx,rz=np.nonzero(ring)
        seal=B.DIRT if y==self.levels[1] else B.STONE
        for dy in (4,3,2,1):
            col=w.vox[rx,y-dy,rz]; w.vox[rx,y-dy,rz]=np.where(B.SOLID[col]|(col==B.WATER),col,seal)
        if y==self.levels[1]:
            # Sandy beach on most of the natural shoreline.
            beach=N.hash_unit(rx//2,rz//2,O.stable_seed(*identity,'beach')%100003)<.7
            col=w.vox[rx,y-1,rz]
            w.vox[rx,y-1,rz]=np.where(beach&np.isin(col,(B.GRASS,B.DIRT,B.STONE)),B.SAND,col)
            self._surface_grade(w,water,y-1,rim=1.5)
        self._canal_paths=getattr(self,'_canal_paths',[])+[[(int(ax),int(az)) for ax,az in legs]]
        return [np.array((ax,y,az),float) for ax,az in legs]

    def _trail_network(self,w,layer,hub,positions):
        """Branching regional trails. From the hub, each remaining site (the
        one nearest the network first) joins the closest point on the trails
        already laid with a gently curving spur, so trails merge and branch
        instead of each running separately to the hub."""
        sx,sz=w.size_x,w.size_z; hub=np.asarray(hub,float)
        nodes=list(dict.fromkeys(tuple(int(v) for v in p) for p in positions))
        nodes=[p for p in nodes if np.hypot(p[0]-hub[0],p[1]-hub[1])>=3]
        segments=[]
        def nearest(p):
            p=np.asarray(p,float)
            if not segments: return float(np.hypot(*(p-hub))),hub
            best=(1e18,hub)
            for a,b in segments:
                d=b-a; t=float(np.clip(((p-a)@d)/max(float(d@d),1e-9),0,1)); q=a+t*d; n=float(np.hypot(*(p-q)))
                if n<best[0]: best=(n,q)
            return best
        while nodes:
            found=[nearest(p) for p in nodes]
            k=int(np.argmin([d for d,_ in found])); p=np.asarray(nodes.pop(k),float); dist,q=found[k]
            if dist<3: continue
            chord=q-p; length=float(np.hypot(*chord)); normal=np.array((-chord[1],chord[0]))/max(length,1e-9)
            bow=(O.unit('trail-bow',self._seed,self.task,layer,int(p[0]),int(p[1]))-.5)*(.3 if layer==1 else .1)*length
            via=np.clip(p+.5*chord+bow*normal,12,np.array((sx,sz))-13)
            points=[tuple(int(v) for v in p),tuple(int(v) for v in via),tuple(int(round(v)) for v in q)]
            self._road(w,layer,points,meander=3.0 if layer==1 else 1.0,radius=20.0 if layer==1 else 8.0)
            segments+=[(p,via),(via,q)]

    def _refill_base_channel(self,w,water,points,width,y):
        """The reference raft setup digs a straight two-deep channel along the
        authored line. Where the meandering river leaves that line, refill the
        leftover channel (water at y-1 and y-2 but not y-3, outside this river)
        to the height of the surrounding ground, so no unsealed straight trench
        remains. Other rivers' three-deep lanes and shallows are untouched.
        Within 30 m of the dock the channel is kept as a dock basin (the craft's
        staging trial runs along it) and returned so it is sealed with the river."""
        from scipy.ndimage import distance_transform_edt
        p=np.asarray([(q[0],q[2]) for q in points],float)
        pad=width+6
        lo=np.maximum(np.floor(p.min(0)-pad).astype(int),0)
        hi=np.minimum(np.ceil(p.max(0)+pad).astype(int)+1,(w.size_x,w.size_z))
        vox=w.vox[lo[0]:hi[0],:,lo[1]:hi[1]]
        basin=np.zeros(water.shape,bool)
        old=(vox[:,y-1,:]==B.WATER)&(vox[:,y-2,:]==B.WATER)&(vox[:,y-3,:]!=B.WATER)&~water[lo[0]:hi[0],lo[1]:hi[1]]
        gx,gz=np.meshgrid(np.arange(lo[0],hi[0])+.5,np.arange(lo[1],hi[1])+.5,indexing='ij')
        dock=np.hypot(gx-p[0,0],gz-p[0,1])<=30
        basin[lo[0]:hi[0],lo[1]:hi[1]]=old&dock; old&=~dock
        if not old.any(): return basin
        band=B.SOLID[vox[:,30:83,:]]
        top=np.where(band.any(axis=1),30+band.shape[1]-1-np.argmax(band[:,::-1,:],axis=1),y-1)
        source=~old&~(vox[:,y-1,:]==B.WATER)
        if not source.any(): return basin
        _,(ix,iz)=distance_transform_edt(~source,return_indices=True)
        ground=np.maximum(top[ix,iz],y-1)
        for i,j in np.argwhere(old):
            g=int(ground[i,j])
            vox[i,y-2:g,j]=B.DIRT; vox[i,g,j]=B.GRASS; vox[i,g+1:g+6,j]=np.where(vox[i,g+1:g+6,j]==B.WATER,B.AIR,vox[i,g+1:g+6,j])
        return basin

    def _meander_legs(self,legs,identity,spacing=40.0,keep=25.0):
        """Raft lane for a long straight waterway as a staircase of axis-aligned
        legs that jogs 9-12 m to alternate sides about every ``spacing`` metres,
        so the river around it meanders. The first and last ``keep`` metres of
        each leg stay on the authored line (docks, yards and aprons are
        unchanged); a jog is skipped where it would come near a site, target,
        dock or protected structure. The craft still steers one axis at a time."""
        avoid=[np.asarray(m.pos,float)[[0,2]] for m in self.milestones]
        avoid+=[np.asarray(o['pos'],float)[[0,2]] for o in self.targets.values()]
        for r in self.rafts.values():
            avoid+=[np.asarray(r[k],float)[[0,2]] for k in ('pos','destination','receiver') if k in r]
        boxes=self.__dict__.get('_protect_boxes',[])
        def clear(p):
            if any(np.hypot(*(p-q))<24 for q in avoid): return False
            return not any(lo[0]-12<=p[0]<=hi[0]+12 and lo[2]-12<=p[1]<=hi[2]+12 for lo,hi in boxes)
        out=[np.asarray(legs[0],float)]
        for k,(a,b) in enumerate(zip(legs,legs[1:])):
            a,b=np.asarray(a,float),np.asarray(b,float)
            axis=0 if a[1]==b[1] else 1; side=1-axis
            length=abs(b[axis]-a[axis]); sign=np.sign(b[axis]-a[axis])
            if length<2*keep+spacing:
                out.append(b); continue
            # Bends are keyed to the line and start of the leg (not the whole
            # route), at a fixed spacing from the start, so rafts sharing a leg
            # share one river and legs that overlap share their first bends.
            line=('skylands-lane',self._seed,self.task,axis,int(sign),float(a[0]),float(a[1]))
            step=spacing*(.8+.5*O.unit(*line,'spacing'))
            stops=[keep+i*step for i in range(int((length-2*keep)//step)+1)]
            if len(stops)<2:
                out.append(b); continue
            count=len(stops)-1
            def at(along,offset):
                # Whole-cell steps from the leg start, as the craft moves.
                p=a.copy(); p[axis]=a[axis]+sign*round(along); p[side]=a[side]+round(offset); return p
            offset=0.0
            for i in range(count):
                # Irregular bends: varied size, and sometimes the river keeps to
                # the same side for a longer reach.
                flip=O.unit(*line,'flip',i)<.72 or offset==0
                jog=(6+7*O.unit(*line,'jog',i))*(-np.sign(offset) if flip and offset else (np.sign(offset) or 1))
                lo,hi=at(stops[i],jog),at(stops[i+1],jog)
                if not (clear(lo) and clear(hi) and clear((lo+hi)/2)): jog=0.0
                if jog!=offset: out+=[at(stops[i],offset),at(stops[i],jog)]
                offset=jog
            if offset: out+=[at(stops[-1],offset),at(stops[-1],0.0)]
            out.append(b)
        # Drop repeated and collinear points.
        clean=[out[0]]
        for p in out[1:]:
            if np.allclose(p,clean[-1]): continue
            if len(clean)>=2:
                d1=clean[-1]-clean[-2]; d2=p-clean[-1]
                if abs(d1[0]*d2[1]-d1[1]*d2[0])<1e-9 and d1@d2>0: clean[-1]=p; continue
            clean.append(p)
        return [tuple(map(float,p)) for p in clean]

    def _natural_shore(self,w,mask,legs,identity,round_bends=8,margin=(1.0,5.0)):
        """Shallow river margin around a surface raft lane: the lane's inner
        corners are rounded off (so jogs read as smooth bends) and an irregular
        margin of ``margin`` metres is added. It narrows to the authored lane
        near the docks at either end; trails, protected shafts and built blocks
        are left as they are."""
        from scipy.ndimage import binary_closing, distance_transform_edt
        from ..core.noise import fbm2
        idx=np.argwhere(mask); pad=round_bends+int(margin[1])+4
        lo=np.maximum(idx.min(0)-pad,0); hi=np.minimum(idx.max(0)+pad+1,mask.shape)
        sub=mask[lo[0]:hi[0],lo[1]:hi[1]]
        r=round_bends; yy,xx=np.mgrid[-r:r+1,-r:r+1]
        smooth=binary_closing(np.pad(sub,r),structure=(xx*xx+yy*yy)<=r*r)[r:-r,r:-r]|sub
        gx,gz=np.meshgrid(np.arange(lo[0],hi[0],dtype=float)+.5,np.arange(lo[1],hi[1],dtype=float)+.5,indexing='ij')
        seed=O.stable_seed(*identity,'shore')%100003
        extra=margin[0]+(margin[1]-margin[0])*np.clip(.5+.6*fbm2(gx/15.,gz/15.,seed,octaves=3),0,1)
        start,end=np.asarray(legs[0],float),np.asarray(legs[-1],float)
        dock=np.minimum(np.hypot(gx-start[0],gz-start[1]),np.hypot(gx-end[0],gz-end[1]))
        fade=np.clip((dock-11)/14,0,1)
        river=(smooth&(fade>0))|(distance_transform_edt(~smooth)<=extra*fade)
        shore=np.zeros(mask.shape,bool); shore[lo[0]:hi[0],lo[1]:hi[1]]=river
        shore&=~mask
        shore&=~binary_dilation(self._surface_roads,iterations=1)
        shore&=~self._protected(1)
        xs,zs=np.nonzero(shore); y=self.levels[1]
        column=w.vox[xs,y-1:y+5,zs]
        keep=(O.CUTTABLE[column]|(column==B.AIR)|(column==B.WATER)).all(axis=1)
        shore[xs[~keep],zs[~keep]]=False
        return shore

    def _road(self,w,layer,points,**organic):
        if self.organic_world: return self._organic_road(w,layer,points,**organic)
        y=self.levels[layer]
        for (ax,az),(bx,bz) in zip(points,points[1:]):
            # Manhattan segments retain explicit traversable geometry.
            for (x0,z0),(x1,z1) in [((ax,az),(bx,az)),((bx,az),(bx,bz))]:
                lo=(min(x0,x1)-2,y-1,min(z0,z1)-2); hi=(max(x0,x1)+2,y-1,max(z0,z1)+2)
                w.fill_box(lo,hi,B.BRICK)
                w.fill_box((lo[0],y,lo[2]),(hi[0],y+2,hi[2]),B.AIR)
                self._road_mask[lo[0]:hi[0]+1,lo[2]:hi[2]+1]=True

    def _author_vertical_connections(self,w):
        x,z=self.vertical_xy
        w.fill_box((x-2,8,z-3),(x+4,110,z+3),B.AIR)
        # Three genuine Ascend steps. Offset holes provide safe reverse travel.
        for floor,left,right,hole in ((27,-13,4,-11),(71,-13,4,-11),(95,-9,4,-7),(111,-5,10,-3)):
            w.fill_box((x+left,floor,z-4),(x+right,floor,z+4),B.BRICK)
            w.fill_box((x+hole,floor,z-1),(x+hole+2,floor,z+1),B.AIR)
        for y in self.levels: self._road(w,self.levels.index(y),[(x,z),self.point(60,180)])
        # Restore the intended holes after road construction.
        for floor,hole in ((27,-11),(71,-11),(95,-7),(111,-3)):
            w.fill_box((x+hole,floor,z-1),(x+hole+2,floor,z+1),B.AIR)
        # Deep ascent needs an intermediate ceiling within the native 32 m reach.
        w.fill_box((x-2,27,z-3),(x+4,27,z+3),B.BRICK)
        dx,dz=self.descent_xy
        w.fill_box((dx-3,8,dz-3),(dx+3,57,dz+3),B.AIR)
        w.fill_box((dx-4,7,dz-4),(dx+4,7,dz+4),B.BASALT)
        # Separate sky descent aperture lands on the surface, above solid ground.
        w.fill_box((dx+15,49,dz-4),(dx+23,114,dz+4),B.AIR)
        w.fill_box((dx+14,47,dz-5),(dx+24,47,dz+5),B.BRICK)
        w.fill_box((dx+14,48,dz-5),(dx+24,50,dz+5),B.AIR)
        self._road(w,2,[(dx+12,dz),(dx+12,dz+10),self.point(60,180)])
        # The sky hoist has its own service ascent chain, not a return to a hub.
        if self.task=='a':
            hx,hz=self.point(460,520)
            w.fill_box((hx-2,54,hz-2),(hx+2,110,hz+2),B.AIR)
            for floor in (77,101,111):
                w.fill_box((hx-2,floor,hz-2),(hx+2,floor,hz+2),B.BRICK)
            self._road(w,2,[(hx,hz),self.point(60,180)])

    def _reset_mechanics(self):
        super()._reset_mechanics()
        # Remove the testbed's universal maintenance lift. Regional physical
        # ascent/descent routes and the task's actual freight lifts do the work.
        self._clear_deck(self.movers['transit']); self.movers.pop('transit')
        for tid in [self.transit_target,*self.lift_calls]: self.targets.pop(tid,None)
        self.cache_bindings={}; self.return_choice=None; self.return_rafts=set()
        self._drive_events=[]
        if self.task=='a':
            m=self.by_id['A06']; self.return_rafts.add('A06')
            self.cache_contents={}
            for tid in [k for k,o in self.targets.items() if o['role']=='cache']: self.targets.pop(tid)
            for k in (1,2):
                x,z=self.point(80,500+20*k)
                alt=Site(f'A06_dock{k}',m.title,'stage',1,x,z,elevation=48)
                self._pad(self.world,alt.pos,12); self._road(self.world,1,[(x,z),self.point(60,180)])
                self._add_raft(alt,alt.id); self.return_rafts.add(alt.id)
                cid=alt.id+'_ballast'; self._add_cargo(cid,alt.pos+(-2,0,4),240)
                self.rafts[alt.id]['cargo_id']=cid
            dest=self.by_id['A14'].pos+(-16,0,4)
            for rid in sorted(self.return_rafts):
                r=self.rafts[rid]; r['destination']=dest.copy(); r['receiver']=dest+(5,0,0)
                r['route']=[np.array((r['pos'][0],48,dest[2])),dest.copy()]
                dug=self._water_route([r['pos'],*r['route']],width=3)
                if dug: r['route']=[q.copy() for q in dug[1:]]
            # Solid receiver apron connects all return choices to the hoist.
            self._pad(self.world,self.by_id['A14'].pos+(8,0,4),5)
            self._author_vault()
        else:
            for tid in [k for k,o in self.targets.items() if o['role']=='cache']: self.targets.pop(tid)
            self.cache_contents={k:[] for k in range(6)}
            for k,(layer,item,x,z) in enumerate([(0,'lamp_kit',80,100),(0,'lamp_kit',120,140),(0,'lamp_kit',160,100),
                                                (1,'lock_coupling',70,110),(1,'lock_coupling',90,130),(1,'lock_coupling',110,110)]):
                px,pz=self.point(x,z); pos=np.array((px,self.levels[layer],pz),float)
                self._pad(self.world,pos,7)
                self._target('regional_reserves','cache',pos+(0,1,0),option=k,item=item,layer=layer)
            r=self.rafts['B23']
            self.refuge_docks=[np.array((*self.point(510,580),),float),np.array((*self.point(450,630),),float)]
            self.refuge_docks=[np.array((p[0],48,p[1]+4)) for p in self.refuge_docks]
            self.refuge_receivers=[p+(0,0,5) for p in self.refuge_docks]
            self.refuge_routes=[]
            for dest in self.refuge_docks:
                dug=self._water_route([r['pos'],np.array((dest[0],48,r['pos'][2])),dest],width=6)
                self.refuge_routes.append([q.copy() for q in dug[1:]] if dug else [np.array((dest[0],48,r['pos'][2])),dest.copy()])
                self._pad(self.world,dest+(0,0,6),4)
            r['destination']=self.refuge_docks[self.destination].copy()
            r['receiver']=self.refuge_receivers[self.destination].copy()
            r['route']=[q.copy() for q in self.refuge_routes[self.destination]]
        self._connect_logistics()
        for mv in self.movers.values():
            self._clear_deck(mv); self._write_deck(mv)
        self.architecture=[]; self._settlement_mask=None
        if self.organic_world:
            self._populate_organic()
        self.world.recompute_heightmaps()

    def _populate_organic(self):
        w=self.world
        for r in self.rafts.values(): self._clear_deck(r)
        # Hamlets, farmsteads and working buildings: collision-backed, off the
        # trails, authored before water settles and before groves and creatures.
        from ..world.settlements import plan
        self.architecture,self._settlement_mask=plan(self)
        self._author_relief()
        water_band=(1,60)   # cavern and surface canals; no water on the sky layer
        O.settle_water(w,band=water_band)
        O.assert_water_contained(w,band=water_band)
        for r in self.rafts.values():
            for q in [r['pos'],r['destination'],*r['route']]:
                x,y,z=np.rint(q).astype(int)
                if w.get(x,y-1,z)!=B.WATER: raise RuntimeError(f'Raft waterway lost water at {(x,y,z)}')
            if all(p is not None for p in r['parts'][:2]): self._write_deck(r)
        scenery=self._dress_surface()
        planted=self._plant_trail_orchards()
        surface=O.surface_top(w,30,82)
        course=self._road_mask|self._settlement_mask
        course|=binary_dilation(w.vox[:,self.levels[1]-1,:]==B.WATER,iterations=2)
        heights=np.where(surface>=0,surface,-1)
        wet=(w.vox[np.arange(w.size_x)[:,None],np.clip(heights+1,0,w.size_y-1),np.arange(w.size_z)[None,:]]==B.WATER)
        heights=np.where(wet,-1,heights)
        # Tree crowns and trunks are not ground to stand on.
        crown=np.isin(w.vox[np.arange(w.size_x)[:,None],np.clip(heights,0,w.size_y-1),np.arange(w.size_z)[None,:]],(B.LEAVES,B.LOG))
        heights=np.where(crown|self._settlement_mask,-1,heights)
        spawns=populate(w,course,('skylands-campaign',self._seed,self.task),sea_level=40,heights=heights,
                        monster_clearance=60.0,scale=4.0,
                        wildlife_band=(8.0,40.0),wildlife_scale=8 if self.task=='a' else 10)   # beyond night sight plus roaming
        x,z=self.point(90,90)
        spawns.insert(0,dict(kind='horse',pos=(x+3.5,48.0,z+3.5),home=(x+3.5,z+3.5),roam=1.5,leash=3.0))
        for spawn in spawns: self.npcs.spawn(**spawn)
        band=np.zeros(w.size_y,bool); band[40:80]=True
        # Scenery groves are wild trees, not apple trees: the orchard keeps the
        # same share of the original trees, so apple supply is unchanged.
        region=np.ones((w.size_x,w.size_z),bool)
        for x,z in scenery: region[x,z]=False
        self.orchard=Orchard.from_world(w,self._seed,band=band,include=planted,region=region)

    def _author_relief(self):
        """Ridges, massifs, mesas, lakes and caves on the surface, rising away
        from every trail, site, waterway, settlement and shaft (see
        owb/world/relief.py). Sky-island glide corridors stay below 57 m."""
        from scipy.ndimage import distance_transform_edt
        from ..world import relief as R
        from ..world.settlements import reserved_mask
        w=self.world; identity=('skylands-campaign',self._seed,self.task)
        keep=reserved_mask(self,site_radius=16)|self._road_mask
        keep|=binary_dilation(self._settlement_mask,iterations=6)
        keep|=binary_dilation((w.vox[:,30:83,:]==B.WATER).any(axis=1),iterations=4)
        keep|=binary_dilation(self._protected(1),iterations=6)
        lakes,made=R.basin_lakes(w,keep,identity,count=2 if self.task=='a' else 3,level=self.levels[1]-2,ylo=30,yhi=82)
        keep|=binary_dilation(lakes,iterations=2)
        # Glide corridors: full reach around sky islands, a narrower margin
        # along the sky layer's stone bridges.
        from scipy.ndimage import binary_opening
        sky=B.SOLID[w.vox[:,self.levels[2]-1,:]]
        islands=binary_opening(sky,structure=np.ones((15,15),bool))
        glide=(distance_transform_edt(~islands)<135)|(distance_transform_edt(~sky)<70)
        T,G,mesa=R.raise_ground(w,keep,identity,ylo=30,yhi=82,cap=80,low_cap=glide,low_cap_value=57,
                                wavelength=self.geography_scale)
        mouths=R.caves(w,T,G,keep,identity,count=4 if self.task=='a' else 6,ylo=40)
        from ..world import landmarks as Lm
        castle=Lm.place_castle(self,T,G,keep,identity,level=self.levels[1],low_cap=glide)
        camps=Lm.place_camps(self,T,G,keep,self._road_mask,identity,count=6 if self.task=='a' else 9,
                             level=self.levels[1],avoid=binary_dilation(self._settlement_mask,iterations=4)|lakes)
        for rec in ([castle] if castle else [])+camps:
            self.architecture.append(rec)
            x,_,z=rec['pos']; r=Lm.CASTLE_FOOT+2 if rec['kind']=='castle' else 8
            self._settlement_mask[max(0,x-r):x+r+1,max(0,z-r):z+r+1]=True
        raised=T>G
        self.relief_summary=dict(raised_fraction=float(raised.mean()),max_height=int(T.max()),
            relief_above_trails=int(T.max())-self.levels[1],lakes=[list(map(int,c)) for c in made],caves=mouths,
            castle=castle['pos'] if castle else None,camps=len(camps),
            mesa_fraction=float((mesa>.5).mean()))
        w.meta['relief']=self.relief_summary

    def _dress_surface(self):
        """Surface scenery on natural ground, placed after every trail, canal
        and pad is authored: groves of trees, bushes and mossy boulders, clear
        of trails, water, sites, fixtures and the vertical shafts. Trees left
        floating by later trail or canal cuts are removed first. Returns the
        grove trunk cells. Placement uses a generator seeded from the world
        identity (never the simulation RNG)."""
        from scipy.ndimage import distance_transform_edt, label
        w=self.world; vox=w.vox; level=self.levels[1]
        ylo,yhi=level-14,self.levels[2]-12
        # 1. Whole trees whose trunk no longer stands on solid ground.
        tree=np.isin(vox[:,ylo:yhi,:],(B.LOG,B.LEAVES))
        labels,count=label(tree,structure=np.ones((3,3,3),bool))
        if count:
            log=vox[:,ylo:yhi,:]==B.LOG
            under=np.zeros_like(log); under[:,1:,:]=B.SOLID[vox[:,ylo:yhi-1,:]]&~np.isin(vox[:,ylo:yhi-1,:],(B.LOG,B.LEAVES))
            rooted=np.unique(labels[log&under]); keep=np.zeros(count+1,bool); keep[rooted]=True; keep[0]=True
            vox[:,ylo:yhi,:][~keep[labels]]=B.AIR
        # 2. Where scenery may go.
        top=O.surface_top(w,30,82); xs_,zs_=np.indices(top.shape)
        t=np.clip(top,0,w.size_y-1)
        ground=vox[xs_,t,zs_]
        headroom=~B.SOLID[vox[xs_,np.clip(t+1,0,w.size_y-1),zs_]]&(vox[xs_,np.clip(t+1,0,w.size_y-1),zs_]!=B.WATER)
        ok=(top>=36)&(top<=80)&np.isin(ground,(B.GRASS,B.DIRT))&headroom
        off_road=distance_transform_edt(~self._surface_roads)
        wet=(vox[:,30:82,:]==B.WATER).any(axis=1)
        off_water=distance_transform_edt(~wet)
        ok&=(off_road>=4.5)&(off_water>=4)
        avoid=[np.asarray(m.pos,float)[[0,2]] for m in self.milestones]
        avoid+=[np.asarray(o['pos'],float)[[0,2]] for o in self.targets.values()]
        avoid+=[np.asarray(r['pos'],float)[[0,2]] for r in self.rafts.values()]
        avoid+=[np.asarray(r['destination'],float)[[0,2]] for r in self.rafts.values()]
        avoid+=[np.asarray(self.vertical_xy,float),np.asarray(self.descent_xy,float),
                np.asarray(self.descent_xy,float)+(19,0),np.asarray(self.point(90,90),float)]
        for a in avoid:
            x0,z0=int(a[0]),int(a[1]); ok[max(0,x0-14):x0+15,max(0,z0-14):z0+15]=False
        for lo,hi in self.__dict__.get('_protect_boxes',[]):
            ok[max(0,lo[0]-4):hi[0]+5,max(0,lo[2]-4):hi[2]+5]=False
        ok&=~binary_dilation(self._protected(1),iterations=4)
        if self.__dict__.get('_settlement_mask') is not None: ok&=~self._settlement_mask
        rng=np.random.default_rng(O.stable_seed('skylands-scenery',self._seed,self.task))
        sx,sz=top.shape
        def free(x,z,r):
            return 0<=x-r and x+r<sx and 0<=z-r and z+r<sz and ok[x-r:x+r+1,z-r:z+r+1].all()
        # Scenery concentrates where travellers see it: within ~45 m of trails,
        # with a sparse scatter beyond.
        cells=np.argwhere(ok); near=off_road[cells[:,0],cells[:,1]]
        weight=np.where(near<=45,1.0,.06); weight/=weight.sum()
        def pick(count):
            return cells[rng.choice(len(cells),size=min(count,len(cells)),replace=False,p=weight)]
        def crown(x,cy,z,r0,key):
            dx,dy,dz=np.meshgrid(np.arange(-3,4),np.arange(-2,3),np.arange(-3,4),indexing='ij')
            jitter=N.hash_unit3(x+dx,cy+dy,z+dz,O.stable_seed(key,self._seed)%100003)
            inside=np.sqrt(dx*dx+dz*dz+(1.3*dy)**2)<=r0+.6*jitter
            region=vox[x-3:x+4,cy-2:cy+3,z-3:z+4]
            region[inside&(region==B.AIR)]=B.LEAVES
        # 3. Groves: clustered trees of varied height with irregular crowns.
        trunks=[]
        groves=int(240*self.geography_scale)
        for gx,gz in pick(groves):
            spread=float(rng.uniform(5,13)); n=int(rng.integers(5,15))
            for _ in range(n):
                x=int(round(gx+rng.normal(0,spread))); z=int(round(gz+rng.normal(0,spread)))
                if not free(x,z,3) or off_road[x,z]<6.5: continue
                y=int(top[x,z])+1; height=int(rng.integers(5,10))
                # Clear trunk column and crown space (slopes may rise around the trunk).
                if B.SOLID[vox[x,y:y+height,z]].any() or B.SOLID[vox[x-3:x+4,y+height-3:y+height+3,z-3:z+4]].any(): continue
                vox[x,y:y+height,z]=B.LOG
                crown(x,y+height-1,z,2.2+1.2*float(rng.random()),'grove-crown')
                ok[x-2:x+3,z-2:z+3]=False; trunks.append((x,z))
        # 4. Bushes: around groves and scattered through the meadows.
        bushes=0
        anchors=trunks+[tuple(c) for c in pick(int(1400*self.geography_scale))]
        for ax,az in anchors:
            for _ in range(int(rng.integers(1,4))):
                x=int(round(ax+rng.normal(0,4))); z=int(round(az+rng.normal(0,4)))
                if not (0<=x<sx and 0<=z<sz and ok[x,z]): continue
                y=int(top[x,z])+1
                if vox[x,y,z]!=B.AIR: continue
                vox[x,y,z]=B.BUSH; ok[x,z]=False; bushes+=1
        # 5. Mossy boulders half sunk into the ground.
        boulders=0
        for x,z in pick(int(260*self.geography_scale)):
            x,z=int(x),int(z); r=float(rng.uniform(1.0,2.6))
            R=int(np.ceil(r))+1
            if not free(x,z,R): continue
            y=int(top[x,z])
            dx,dy,dz=np.meshgrid(np.arange(-R,R+1),np.arange(-1,R+1),np.arange(-R,R+1),indexing='ij')
            jitter=N.hash_unit3(x+dx,y+dy,z+dz,O.stable_seed('boulder',self._seed)%100003)
            inside=np.sqrt(dx*dx+dz*dz*1.3+(1.5*dy)**2)<=r*(.8+.4*jitter)
            region=vox[x-R:x+R+1,y-1:y+R+1,z-R:z+R+1]
            region[inside]=np.where(dy>=r*.5,B.MOSS_STONE,B.STONE)[inside]
            ok[x-R:x+R+1,z-R:z+R+1]=False; boulders+=1
        self.scenery_summary=dict(grove_trees=len(trunks),bushes=bushes,boulders=boulders)
        return set(trunks)

    def _plant_trail_orchards(self,spacing=TRAIL_ORCHARD_SPACING):
        """Plant apple trees on the natural ground just beside surface trails,
        off the walkable trail itself and clear of sites, fixtures and water,
        so travellers can shake apples loose. Placement uses a generator seeded
        from the world identity."""
        w=self.world; vox=w.vox; level=self.levels[1]
        from scipy.ndimage import distance_transform_edt
        stand=B.SOLID[vox[:,level-1,:]]&~B.SOLID[vox[:,level,:]]&~B.SOLID[vox[:,level+1,:]]
        road=self._road_mask&stand
        off=distance_transform_edt(~road)
        top=O.surface_top(w,level-3,level+3)
        ok=(off>=1.9)&(off<=2.6)&(top>=level-2)&(top<=level+1)
        wet=binary_dilation((vox[:,level-6:level+2,:]==B.WATER).any(axis=1),iterations=4)
        ok&=~wet
        avoid=[np.asarray(m.pos,float)[[0,2]] for m in self.milestones]
        avoid+=[np.asarray(t['pos'],float)[[0,2]] for t in self.targets.values()]
        avoid+=[np.asarray(r['pos'],float)[[0,2]] for r in self.rafts.values()]
        for lo,hi in self.__dict__.get('_protect_boxes',[]):
            ok[max(0,lo[0]-3):hi[0]+4,max(0,lo[2]-3):hi[2]+4]=False
        if self.__dict__.get('_settlement_mask') is not None: ok&=~self._settlement_mask
        cells=np.argwhere(ok)
        rng=np.random.default_rng(O.stable_seed('trail-orchard',self.task,self._seed))
        chosen=[]
        for i in rng.permutation(len(cells)):
            x,z=map(int,cells[i]); y=int(top[x,z])+1
            if any(np.hypot(x-a[0],z-a[1])<10 for a in avoid): continue
            if any(np.hypot(x-c[0],z-c[1])<spacing for c in chosen): continue
            if vox[x,y-1,z] not in (B.GRASS,B.DIRT,B.SAND,B.GRAVEL,B.MUD,B.STONE): continue
            height=4+int(2*O.unit('trail-orchard',self._seed,x,z))
            crown=vox[x-2:x+3,y+2:y+height+3,z-2:z+3]
            if B.SOLID[crown].any() or B.SOLID[vox[x,y:y+height,z]].any(): continue
            vox[x,y-1,z]=B.GRASS
            vox[x,y:y+height,z]=B.LOG
            # An irregular, rounded crown keyed to the tree position.
            cy=y+height
            for dx in range(-2,3):
                for dz in range(-2,3):
                    for dy in range(-1,2):
                        r=np.sqrt(dx*dx+dz*dz+(1.4*dy)**2)
                        if r<=2.35+0.5*O.unit('trail-crown',self._seed,x+dx,z+dz,dy) and vox[x+dx,cy+dy,z+dz]==B.AIR:
                            vox[x+dx,cy+dy,z+dz]=B.LEAVES
            chosen.append((x,z))
        return set(chosen)

    def fan_strike(self):
        """Charge a raft by striking its mounted fan: a clean strike lands as the
        previous spin fades (0.5-1.5 s); too soon jams it and wastes charge."""
        a=self.agent
        for rid,r in self.rafts.items():
            fan=r['parts'][2]
            if fan is None or np.linalg.norm(r['pos']+(0,1,0)-a.eye)>3.4: continue
            since=self.elapsed-r.get('spun',-99.)
            if since<FAN_SWEET[0]: gain,quality=FAN_JAM,'jammed'
            elif since<=FAN_SWEET[1]: gain,quality=FAN_GOOD,'clean'
            else: gain,quality=FAN_LATE,'from rest'
            r['charge']=float(np.clip(r['charge']+gain,0,300)); r['spun']=self.elapsed
            return [dict(type='fan_struck',raft=rid,quality=quality,charge=r['charge'])]
        return []

    def _add_raft(self,m,rid):
        if rid=='B23': m=replace(m,z=m.z+8)
        super()._add_raft(m,rid); r=self.rafts[rid]
        r.update(destination=np.array((r['end'],m.y,m.z+4)),receiver=np.array((r['end']+5,m.y,m.z+4)),
                 drive_vector=np.zeros(3),driving=False,origin_pos=r['pos'].copy())
        r['route']=[r['destination'].copy()]
        dug=self._water_route([r['pos'],r['destination']],width=6)
        if dug: r['route']=[q.copy() for q in dug[1:]]
        # Real gathering work across the supply yard, with inspectable variants.
        offsets=((-18,-16),(-28,14),(-14,-27),(-30,-22))
        for (pid,p),(dx,dz) in zip([(pid,p) for pid,p in self.parts.items() if p['owner']==rid],offsets):
            pos=m.pos+(dx,0,dz); p['pos']=pos
            self._pad(self.world,pos,3)
            self._road(self.world,m.layer,[(int(pos[0]),int(pos[2])),(m.x-3,m.z)])
            for target in self.targets.values():
                if target.get('part')==pid: target['pos']=pos+(0,.7,0)
        # Spare compatible hull: several supported assemblies are possible.
        pid=max(self.parts,default=0)+1
        self.parts[pid]=dict(kind='pontoon',pos=m.pos+(-23,0,21),owner=rid,mounted=False,turn=0,capacity=260)
        self._pad(self.world,self.parts[pid]['pos'],3)
        self._road(self.world,m.layer,[(m.x-23,m.z+21),(m.x-3,m.z)])
        self._target(rid,'part',self.parts[pid]['pos']+(0,.7,0),part=pid)

    def _add_reversal(self,m):
        super()._add_reversal(m)
        # The shuttle's real gap must never be paved over by nearby trails.
        self._protect_boxes.append(((m.x+1,m.y-1,m.z-2),(m.x+self.shuttle_span(m)+1,m.y-1,m.z+2)))
        self.world.fill_box((m.x+1,m.y,m.z-2),(m.x+self.shuttle_span(m)+5,m.y+3,m.z+2),B.AIR)
        end=m.x+self.shuttle_span(m)+3
        # A supported service egress from the landing prevents a successful
        # platform crossing from stranding the traveler above uneven terrain.
        self._road(self.world,m.layer,[(end,m.z),(end,m.z+8),(m.x-4,m.z+8),(m.x-4,m.z)])

    def _water_route(self,points,width):
        if self.organic_world: return self._organic_water(points,width)
        y=int(points[0][1])
        for a,b in zip(points,points[1:]):
            ax,az=int(a[0]),int(a[2]); bx,bz=int(b[0]),int(b[2])
            if ax!=bx and az!=bz: raise ValueError('Author orthogonal navigable waterway legs')
            if az==bz:
                lo=(min(ax,bx)-1,y-3,az-width); hi=(max(ax,bx)+1,y-1,az+width)
            else:
                lo=(ax-width,y-3,min(az,bz)-1); hi=(ax+width,y-1,max(az,bz)+1)
            self.world.fill_box(lo,hi,B.WATER)
            self.world.fill_box((lo[0],y,lo[2]),(hi[0],y+4,hi[2]),B.AIR)

    def _author_vault(self):
        m=self.by_id['A10']; x,y,z=m.x,m.y,m.z
        for lo,hi in [((x-10,y,z-10),(x-10,y+6,z+10)),((x+10,y,z-10),(x+10,y+6,z+10)),
                      ((x-10,y,z-10),(x+10,y+6,z-10)),((x-10,y,z+10),(x+10,y+6,z+10))]:
            self.world.fill_box(lo,hi,B.BRICK)
        self.vault_gate=((x-10,y,z-2),(x-10,y+3,z+2))

    def _dispatch(self,command,ingredients,action):
        tid=int(action['target']); obj=self.targets.get(tid); option=int(action['option'])
        for m in self.milestones:
            if 'load_target' in m.state:
                self.targets[m.state['load_target']]['pos']=self.movers[m.state['mover']]['pos']+(0,1,0)
        if obj and obj['role']=='raft' and obj['owner'] in self.rafts: obj['pos']=self.rafts[obj['owner']]['pos']+(0,1,0)
        if obj and command=='haul' and obj['role']=='cargo':
            cid=obj['owner']; c=self.cargo[cid]
            if c['anchored'] or c['parent'] or np.linalg.norm(c['pos']-self.agent.pos)>5:
                return self._advance(self._fail('requested loose cargo is not within tow reach'))
            self.hauling=cid; self.tow_offset=c['pos']-self.agent.pos
            return self._advance([dict(type='cargo_towed',cargo=cid)])
        if obj and command=='inspect' and obj['role'] in ('part','raft'):
            if not self._reachable(obj['pos'],6): return self._advance(self._fail('inspect within local visible reach'))
            if obj['role']=='part':
                p=self.parts[obj['part']]; text=f"{p['kind']}; flotation rating {p['capacity']} kg; orientation {p['turn']}."
            else:
                r=self.rafts[obj['owner']]; mass=160+(self.cargo[r['cargo']]['mass'] if r['cargo'] else 0)
                text=f"Total load {mass} kg; charge {r['charge']}; supported {self.craft_valid(r)}."
            self.last_text=text; return self._advance([dict(type='clue_exposed',site=obj['owner'],text=text)])
        if obj and command=='cache' and obj['role']=='cache':
            if not self._reachable(obj['pos']): return self._advance(self._fail('cache not reachable'))
            item=obj['item']; key=obj['option']
            if not self.satchel[item]: return self._advance(self._fail('reserve item not carried'))
            self.satchel[item]-=1; self.cache_contents[key].append(item); self.cache_bindings[item]=key
            self.by_id['B09'].state['done']={'lamp_kit','lock_coupling'}<=self.cache_bindings.keys()
            return self._advance([dict(type='reserves_cached',cache=key,item=item,layer=obj['layer'])])
        if obj and obj['owner'] in self.rafts:
            rid=obj['owner']; r=self.rafts[rid]
            if command=='drive':
                if not self._reachable(obj['pos'],6) or not self.on_deck(r) or not self.craft_valid(r):
                    return self._advance(self._fail('board a charged, supported craft before driving'))
                v=np.sign(action['build_move']).astype(float); v[1]=0
                if np.count_nonzero(v)>1: return self._advance(self._fail('one steering axis per decision'))
                r['drive_vector']=v if option!=2 else np.zeros(3); r['driving']=bool(np.any(r['drive_vector']))
                r['moored']=False
                return self._advance([])
            if command=='use' and rid in self.return_rafts:
                if not self._reachable(obj['pos']): return self._advance(self._fail('moor within reach'))
                if not self.craft_valid(r) or r['travel']<8 or np.linalg.norm(r['pos']-r['origin_pos'])>.1:
                    return self._advance(self._fail('complete the loaded trial and return to the chosen dock'))
                r['drive_vector'][:]=0; r['driving']=False; r['moored']=True; self.return_choice=rid
                self.by_id['A06'].state.update(done=True,raft=rid); self.by_id['A13'].state['raft']=rid
                return self._advance([dict(type='raft_staged',raft=rid,position=r['pos'].tolist())])
            if command=='unload':
                if not self._reachable(obj['pos'],6) or not r['cargo']: return self._advance(self._fail('loaded craft not in reach'))
                cid=r['cargo']; c=self.cargo[cid]
                staging=r['moored'] and np.linalg.norm(r['pos']-r['origin_pos'])<.1
                destination=self.refuge_docks[option] if rid=='B23' and option<2 else r['destination']
                if not staging and np.linalg.norm(r['pos']-destination)>.1: return self._advance(self._fail('reach the actual receiving dock before unloading'))
                receiver=r['origin_pos']+(-3,0,0) if staging else (self.refuge_receivers[option] if rid=='B23' else r['receiver'])
                c.update(parent=None,pos=receiver.copy()); r['cargo']=None; r['drive_vector'][:]=0; r['driving']=False
                if rid in self.by_id and self.by_id[rid].kind=='raft': self.by_id[rid].state['done']=True
                if rid in self.return_rafts and cid=='core': self.by_id['A13'].state['done']=True
                if rid=='B23':
                    c['delivered_to']=option
                    if option==self.destination:
                        c['anchored']=True; c['parent']='refuge'; self.by_id['B24'].state['done']=True
                return self._advance([dict(type='cargo_delivered',cargo=cid,raft=rid,receiver=receiver.tolist())])
        events=super()._dispatch(command,ingredients,action)
        if command=='mount' and obj and obj['owner'] in self.rafts and option==3 and any(e['type']=='part_mounted' for e in events):
            # The workshop's spare cell for the first raft arrives nearly flat:
            # the fan must be charged by hand (fan_strike).
            self.rafts[obj['owner']]['charge']=12. if self.organic_world and obj['owner'] in ('A02','B02') else 300.
        if self.organic_world and command=='attack' and any(e['type']=='swing' for e in events):
            events+=self.fan_strike()
        return events

    def _advance(self,events):
        for rid,r in self.rafts.items():
            v=r.get('drive_vector',np.zeros(3))
            if not np.any(v) or not self.on_deck(r) or not self.craft_valid(r): continue
            q=r['pos']+v; x,y,z=np.rint(q).astype(int)
            cell=(x,y-1,z); material=r['under'].get(cell,self.world.get(*cell))
            if material!=B.WATER or self._move_deck(r,q) is None:
                r['drive_vector'][:]=0; r['driving']=False
                events.append(dict(type='motion_blocked',object=rid)); continue
            r['charge']-=.25; r['travel']+=1; r['tested']=r['travel']>=3
            events.append(dict(type='craft_moved',raft=rid,position=q.tolist(),charge=r['charge']))
        return super()._advance(events)

    def _operate(self,m,role,option):
        events=super()._operate(m,role,option)
        if m.id=='A09' and m.state['done']:
            self.world.fill_box(*self.vault_gate,B.AIR)
            events.append(dict(type='vault_door_opened'))
        return events

    def system_operational(self):
        if self.task=='a': return super().system_operational()
        c=self.cargo['relief']
        return (all(self.by_id[mid].state['installed'] and self.by_id[mid].state['stable']>=5 for mid in ('B15','B17','B19'))
            and all(self.cargo[cid]['anchored'] and self.cargo[cid]['parent']==mid and np.linalg.norm(self.cargo[cid]['pos']-self.by_id[mid].pos)<6
                    for cid,mid in (('power_module','B15'),('marsh_coupling','B17')))
            and self.by_id['B19'].state.get('station_part')=='spindle' and self.by_id['B21'].state['installed']
            and self.by_id['B23'].state['installed'] and self.by_id['B24'].state['done']
            and c['intact'] and c['parent']=='refuge' and c.get('delivered_to')==self.destination
            and np.linalg.norm(c['pos']-self.refuge_receivers[self.destination])<.1)

    def state_digest(self):
        base=super().state_digest()
        return hashlib.sha256((base+json.dumps(plain(dict(bindings=self.cache_bindings,return_choice=self.return_choice)),sort_keys=True)).encode()).hexdigest()


    def cache_position(self,k):
        return next(o['pos']-(0,1,0) for o in self.targets.values() if o['role']=='cache' and o['option']==k)

    def refuge_position(self,k): return self.refuge_receivers[k]

    def closure_bounds(self):
        m=self.by_id['B20']
        return (m.x+7,m.y,m.z-2),(m.x+9,m.y+3,m.z+2)

    def _connect_logistics(self):
        """Connect real loading aprons and trails without paving over waterways."""
        from scipy.ndimage import binary_dilation
        endpoints={k:[] for k in range(3)}
        for m in self.milestones:
            endpoints[m.layer].append((m.x-(14 if m.id=='A10' else 4),m.z))
        for r in self.rafts.values():
            layer=self.levels.index(int(r['pos'][1]))
            for pos in (r['origin_pos']+(-4,0,0),r['receiver']):
                self._pad(self.world,pos,3)
                endpoints[layer].append((int(pos[0]),int(pos[2])))
        for mv in self.movers.values():
            if mv['kind']=='freight':
                pos=mv['pos']+(4,mv['top']-mv['pos'][1],0)
                self._pad(self.world,pos,2)
                endpoints[self.levels.index(int(pos[1]))].append((int(pos[0]),int(pos[2])))
        if self.task=='b':
            for p in self.refuge_receivers:
                self._pad(self.world,p,3); endpoints[1].append((int(p[0]),int(p[2])))
            for o in self.targets.values():
                if o['role']=='cache': endpoints[o['layer']].append(tuple(o['pos'][[0,2]].astype(int)))
        # Route each trail around water; 2 m shoulders keep the cradle supported.
        for layer,points in endpoints.items():
            y=self.levels[layer]; water=self.world.vox[:,y-1,:]==B.WATER
            blocked=binary_dilation(water,iterations=3)
            # Avoid authored service walls and the known descent apertures.
            solid=B.SOLID[self.world.vox[:,y,:]] & (self.world.vox[:,y,:]==B.BRICK)
            blocked |= binary_dilation(solid,iterations=3)
            for mv in self.movers.values():
                if mv['kind']=='freight' and mv['pos'][1]<y<=mv['top']:
                    mx,mz=map(int,mv['pos'][[0,2]])
                    blocked[mx-3:mx+4,mz-3:mz+4]=True
            for x,z in [self.descent_xy,(self.descent_xy[0]+19,self.descent_xy[1])]:
                blocked[x-6:x+7,z-6:z+7]=True
            hx,hz=self.point(60,180); goal=(hx,hz)
            for point in points:
                start=tuple(point)
                # The loading apron itself is traversable, even close to water.
                blocked_here=blocked.copy(); px,pz=start
                blocked_here[px-1:px+2,pz-1:pz+2]=False
                queue=[(0,0,start)]; previous={start:None}; cost={start:0}; found=None
                while queue:
                    _,g,q=heapq.heappop(queue)
                    if g!=cost[q]: continue
                    if q==goal: found=q; break
                    for n in ((q[0]-1,q[1]),(q[0]+1,q[1]),(q[0],q[1]-1),(q[0],q[1]+1)):
                        nx,nz=n
                        if not (6<nx<self.world.size_x-6 and 6<nz<self.world.size_z-6) or blocked_here[n]: continue
                        # Existing trails receive preference, preserving regional branches.
                        if self.organic_world: c=g+(1 if self._layer_roads[layer][n] else 3)
                        else: c=g+(1 if self._road_mask[n] else 2)
                        if c<cost.get(n,1e12):
                            cost[n]=c; previous[n]=q
                            heapq.heappush(queue,(c+abs(nx-hx)+abs(nz-hz),c,n))
                if found is None: raise RuntimeError(f'No logistics trail from {layer}:{start}')
                path=[]
                while found is not None: path.append(found); found=previous[found]
                path.reverse()
                if self.organic_world:
                    # Pave only the stretches not already on a trail, each as
                    # any-angle runs joined to the trail at both ends.
                    on=[bool(self._layer_roads[layer][q]) for q in path]; i=0
                    while i<len(path):
                        if on[i]: i+=1; continue
                        j=i
                        while j<len(path) and not on[j]: j+=1
                        run=path[max(0,i-1):min(len(path),j+1)]
                        if len(run)>=2: self._road(self.world,layer,self._pull_string(run,blocked_here))
                        i=j
                    continue
                else:
                    # Group straight stretches to avoid voxel-by-voxel height-map updates.
                    corners=[path[0]]
                    for i in range(1,len(path)-1):
                        if (path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]) != (path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]): corners.append(path[i])
                    corners.append(path[-1])
                self._road(self.world,layer,corners)
        # Preserve freight shafts after the connected road network is authored.
        for mv in self.movers.values():
            if mv['kind']=='freight':
                x,y,z=map(int,mv['pos'])
                self.world.fill_box((x-1,y,z-1),(x+1,mv['top']+3,z+1),B.AIR)
        # Reopen intended holes after nearby trail work.
        dx,dz=self.descent_xy
        self._road(self.world,1,[(dx-6,dz),(dx-6,dz+10),self.point(60,180)])
        self._road(self.world,1,[(dx+19,dz),(dx+19,dz+10),self.point(60,180)])
        self.world.fill_box((dx-3,8,dz-3),(dx+3,57,dz+3),B.AIR)
        self.world.fill_box((dx+15,49,dz-4),(dx+23,114,dz+4),B.AIR)


    @staticmethod
    def _pull_string(path,blocked,step=.35):
        """Any-angle waypoints for a 4-connected grid path: from each waypoint,
        jump to the furthest later path cell whose straight centreline (and a
        cell either side of it) crosses no blocked cell, so the trail keeps the
        A* route's clearance but runs diagonally instead of in L-shaped legs."""
        def clear(a,b):
            a=np.asarray(a,float)+.5; b=np.asarray(b,float)+.5; d=b-a; n=float(np.hypot(*d))
            side=np.array((-d[1],d[0]))/max(n,1e-9)
            for t in np.linspace(0,1,max(2,int(n/step)+1)):
                for o in (0.,1.,-1.):
                    q=np.floor(a+d*t+side*o).astype(int)
                    if blocked[q[0],q[1]]: return False
            return True
        corners=[path[0]]; i=0
        while i<len(path)-1:
            j=len(path)-1
            # Long trails: test a bounded look-ahead, coarse to fine.
            j=min(j,i+160)
            while j>i+1 and not clear(path[i],path[j]): j-=max(1,(j-i)//6)
            corners.append(path[j]); i=j
        return corners

    def static_presentation_meshes(self,batch):
        from ..render.settlements import settlement_meshes
        settlement_meshes(self.architecture,batch)

    def presentation_lights(self):
        lights=super().presentation_lights()
        if self.organic_world:
            from ..world.settlements import lights as settlement_lights
            lights+=settlement_lights(self.architecture)
        if self.satchel['lamp']:
            lights.append(dict(pos=(self.agent.pos+(0,2.4,0)).tolist(),color=[1.,.78,.46],power=1100.,radius=.45))
        return lights
