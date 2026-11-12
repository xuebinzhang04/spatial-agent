"""Render graph-only construction examples; these images are not UE observations."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from cases import City,UPSTREAM,FAMILIES,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args();batch=args.batch.resolve();summary=json.loads((batch/'summary.json').read_text())
    selected=[next(r for r in summary['cases'] if r['family']==f and r['level']==1) for f in FAMILIES]
    canvas=Image.new('RGB',(1000,1010),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/cantarell/Cantarell-Bold.otf',17)
    small=ImageFont.truetype('/usr/share/fonts/cantarell/Cantarell-Bold.otf',13)
    draw.text((24,8),'DeliveryBench construction examples: graph witnesses, UE validation pending',fill='#252525',font=font)
    colors={'start':'#5c4aa0','pickup':'#e59b22','dropoff':'#216aa1','charging':'#339358'}
    notes=['# First construction examples','','These are private graph construction witnesses, not recorded UE trajectories or VLM results.','']
    for index,row in enumerate(selected):
        case=json.loads((batch/row['path']).read_text());city=City(case['map'])
        roads=json.loads((UPSTREAM/'maps'/case['map']/'roads.json').read_text())['roads']
        xs=[r[k]['x'] for r in roads for k in ('start','end')];ys=[r[k]['y'] for r in roads for k in ('start','end')]
        left=20+(index%2)*500;top=48+(index//2)*470;size=420
        scale=min(size/(max(xs)-min(xs)+80),size/(max(ys)-min(ys)+80))
        def point_cm(p):return (left+25+(p[0]/100-min(xs)+40)*scale,top+42+size-(p[1]/100-min(ys)+40)*scale)
        draw.text((left,top),case['family'],fill='#252525',font=font)
        for road in roads:
            a=point_cm([road['start']['x']*100,road['start']['y']*100]);b=point_cm([road['end']['x']*100,road['end']['y']*100])
            draw.line([a,b],fill='#dddddd',width=2)
        nodes={tuple(city.xy(m)):city.node(m) for metas in city.by_type.values() for m in metas}
        legs=[e['graph_leg'] for e in case['private']['serial_witness']['events']]
        if case['private']['charge']:legs.insert(0,case['private']['charge']['approach_graph_leg'])
        for leg in legs:
            a,b=leg['endpoints_cm'];path,_=city.map.shortest_path_nodes(nodes[tuple(a)],nodes[tuple(b)])
            draw.line([point_cm([n.position.x,n.position.y]) for n in path],fill='#73bdc7',width=3)
        def marker(position,kind):
            x,y=point_cm(position);draw.ellipse((x-5,y-5,x+5,y+5),fill=colors[kind],outline='white',width=1)
        marker(case['initial']['position_cm'],'start')
        for order in case['private']['orders']:
            marker(order['pickup']['position_cm'],'pickup');marker(order['dropoff']['position_cm'],'dropoff')
        if case['private']['charge']:marker(case['private']['charge']['station']['position_cm'],'charging')
        draw.text((left,top+450),f"{len(case['private']['orders'])} order(s), {case['private']['serial_witness']['distance_cm']/100:.0f} m graph witness",fill='#555555',font=small)
        notes += ['## '+case['family'],'',
            f"Case: `{case['case_id']}`; map `{case['map']}`; seed `{case['seed']}`; structural level `1`.",
            f"Status: `{case['status']}`. Initial battery: `{case['initial']['battery_pct']:.2f}%`.",
            f"Graph witness: `{case['private']['serial_witness']['distance_cm']/100:.2f} m`, `{case['private']['serial_witness']['duration_s']:.2f} s` under the stated speed/preparation model.",
            f"[Recipe]({row['path']}) · [Public instruction](public/{case['case_id']}.json)",'',case['public']['instruction'],'']
    draw.text((24,992),'Purple: start    Orange: pickup    Blue: destination    Green: charging    Cyan: one graph route',fill='#444444',font=small)
    canvas.save(batch/'construction-examples.png')
    (batch/'EXAMPLES.md').write_text('\n'.join(notes)+'\n')
    (batch/'presentation.json').write_text(json.dumps({'figure':'construction-examples.png','sha256':digest(batch/'construction-examples.png'),
        'source_cases':[r['case_id'] for r in selected],'native_observation':False,'agent_input':False},indent=2)+'\n')
    print(batch/'construction-examples.png')


if __name__=='__main__':main()
