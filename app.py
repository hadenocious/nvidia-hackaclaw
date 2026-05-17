from flask import Flask, render_template, jsonify, request
import requests as req
import pandas as pd
import numpy as np
import threading
import time
import shutil, os
from datetime import datetime


if os.path.exists('data/results.csv'):
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    shutil.copy('data/results.csv', f'data/results{ts}.csv')

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "nemotron3:33b"

ROLL_FULL = "/home/asus/rocket_project/data/thesis/digitized/roll_angular_speed_black.csv"
FINS_FULL = "/home/asus/rocket_project/data/thesis/digitized/fin_angular_velocity_red.csv"
ROLL_ZOOM = "/home/asus/rocket_project/data/thesis/digitized/roll_angular_velocity_black.csv"
FINS_ZOOM = "/home/asus/rocket_project/data/thesis/digitized/fin_angular_speed_red.csv"

PRESETS = {
    "standard": {"name":"Standard 5-Min Coast","goal":"Launch, coast at altitude for 5 minutes, return and land safely.","conditions":["crosswind","sensor_noise"],"coast":300},
    "storm":    {"name":"Storm Launch","goal":"Launch during a storm. Survive 80m/s crosswinds and heavy turbulence. Reach target altitude and return.","conditions":["heavy_crosswind","turbulence","sensor_noise"],"coast":120},
    "sensor":   {"name":"Sensor Failure","goal":"Complete a full flight even though the primary IMU fails at T+30s. Use backup sensors to navigate.","conditions":["imu_failure","sensor_noise"],"coast":180},
    "emergency":{"name":"Emergency Abort","goal":"Engine anomaly at T+20s. Abort ascent immediately and return to launch site safely.","conditions":["engine_anomaly","roll_anomaly","sensor_noise"],"coast":0},
    "precision":{"name":"Precision Landing","goal":"Launch, coast 3 minutes, land within 10 meters of the launch pad.","conditions":[],"coast":180},
    "custom":   {"name":"Custom Mission","goal":"","conditions":[],"coast":300},
}

state = {
    "running":False,"status":"STANDBY","phase":"READY",
    "time":0,"roll_rate":0,"altitude":0,"velocity":0,
    "ai_fin":0,"pid_fin":0,"action":"STANDBY",
    "reasoning":"Configure your mission and hit LAUNCH",
    "decisions":[],"roll_history":[],"ai_history":[],
    "pid_history":[],"time_history":[],"alt_history":[],
    "direction_accuracy":0,"total_decisions":0,
    "coast_elapsed":0,"coast_goal":300,
    "active_conditions":[],"events":[],"mission_result":"",
    "mission_goal":"","tokens_per_sec":0
}

def inject(t, roll, alt, vel, conds):
    active = []
    if "crosswind" in conds and 15<=t<=30:
        roll += np.random.normal(0,60)+80*np.sin((t-15)*np.pi/15)
        active.append("Crosswind 40m/s")
    if "heavy_crosswind" in conds and 10<=t<=60:
        roll += np.random.normal(0,80)+120*np.sin((t-10)*np.pi/25)
        active.append("HEAVY WIND 80m/s")
    if "turbulence" in conds:
        roll += np.random.normal(0,25); alt += np.random.normal(0,6)
        active.append("Turbulence")
    if "sensor_noise" in conds:
        roll += np.random.normal(0,8); alt += np.random.normal(0,3)
    if "imu_failure" in conds and t>=30:
        roll = np.random.uniform(-60,60)
        active.append("IMU FAILURE")
    if "roll_anomaly" in conds and 28<=t<=36:
        roll += 220*np.sin((t-28)*np.pi/8)
        active.append("ROLL ANOMALY 220deg/s")
    if "engine_anomaly" in conds and 18<=t<=22:
        vel *= 0.2; active.append("ENGINE FAULT")
    if "icing" in conds and t>=20:
        roll += np.random.normal(0,12); active.append("Icing")
    return roll, alt, vel, active

def load_data(coast=300):
    rf = pd.read_csv(ROLL_FULL,header=None,names=['t','r']).sort_values('t').drop_duplicates('t')
    rf = rf[(rf['t']>=0)&(rf['t']<=11)]
    rz = pd.read_csv(ROLL_ZOOM,header=None,names=['t','r']).sort_values('t').drop_duplicates('t')
    rz = rz[(rz['t']>=0)&(rz['t']<=2)]
    ff = pd.read_csv(FINS_FULL,header=None,names=['t','f']).sort_values('t').drop_duplicates('t')
    ff = ff[(ff['t']>=0)&(ff['t']<=11)]
    fz = pd.read_csv(FINS_ZOOM,header=None,names=['t','f']).sort_values('t').drop_duplicates('t')
    fz = fz[(fz['t']>=0)&(fz['t']<=2)]

    tz=np.arange(0,2.1,0.5); tf=np.arange(2.5,11.0,1.0)
    roll_r=np.concatenate([np.interp(tz,rz['t'],rz['r']),np.interp(tf,rf['t'],rf['r'])])
    fins_r=np.concatenate([np.interp(tz,fz['t'],fz['f']),np.interp(tf,ff['t'],ff['f'])])
    t_r=np.concatenate([tz,tf])

    ce=12+coast; cd=ce+60; cl=cd+25
    tc=np.arange(12,ce,5); td=np.arange(ce,cd,5); tl=np.arange(cd,cl,2)
    all_t=np.concatenate([t_r,tc,td,tl])
    all_r=np.concatenate([roll_r,np.random.normal(5,12,len(tc)),np.random.normal(0,8,len(td)),np.random.normal(0,4,len(tl))])
    all_f=np.concatenate([fins_r,np.ones(len(tc))*11.5,np.ones(len(td))*8.0,np.zeros(len(tl))])

    all_alt=np.where(all_t<=2,all_t*80,
             np.where(all_t<=8,160+(all_t-2)*55,
             np.where(all_t<=12,490+(all_t-8)*5,
             np.where(all_t<=ce,510+np.random.normal(0,3,len(all_t)),
             np.where(all_t<=cd,510-((all_t-ce)/60)*490,
             np.maximum(20-((all_t-cd)/25)*20,0))))))
    all_alt=np.maximum(all_alt,0)
    all_vel=np.where(all_t<=1.5,all_t*80,
             np.where(all_t<=8,120-(all_t-1.5)*15,
             np.where(all_t<=12,15-(all_t-8)*3,
             np.where(all_t<=ce,np.random.normal(0,2,len(all_t)),
             np.where(all_t<=cd,-(all_t-ce)*1.5,-5)))))
    return list(zip(all_t.tolist(),all_r.tolist(),all_f.tolist(),all_alt.tolist(),all_vel.tolist()))

def get_phase(t,alt,vel,cs,cg):
    ce=(t-cs) if cs else 0
    cr=max(0,cg-ce)
    if t<0.5: return "LAUNCH"
    elif t<8: return "POWERED_ASCENT"
    elif t<12: return "ENGINE_CUTOFF"
    elif cr>0: return "COAST"
    elif alt>100 and vel<-5: return "DESCENT"
    elif alt>30: return "LANDING_BURN"
    else: return "TOUCHDOWN"

SYSTEM_PROMPT = """You are AutoClaw, an AI rocket flight controller. Your goal is NOT to copy a PID controller — your goal is to BEAT it by reasoning smarter.

PHYSICS:
- Roll rate in deg/s. Positive = clockwise spin viewed from behind.
- Fin deflection range: -15 to +15 degrees.
- To stop a POSITIVE roll: use NEGATIVE fin. To stop a NEGATIVE roll: use POSITIVE fin.
- Proportional response: scale your correction to the severity of the roll.
  - |roll| < 20 deg/s  -> small correction (1-3 deg)
  - |roll| 20-60 deg/s -> medium correction (4-8 deg)
  - |roll| > 60 deg/s  -> aggressive correction (9-15 deg)
- DO NOT over-correct: a fin command that is too large causes oscillation.
- If roll is already near zero and settling, reduce fin toward 0 to avoid inducing new spin.
- React to the TREND: if roll is oscillating rapidly, dampen — don't fight each peak.

YOUR EDGE OVER PID:
- PID reacts mechanically. You reason about context.
- Use phase: during POWERED_ASCENT react fast. During COAST, be gentle.
- Use conditions: heavy crosswind needs more aggressive correction.
- Anticipate: if roll just reversed sign, it may be rebounding — don't slam full deflection.

RESPONSE FORMAT — follow exactly, no extra text:
FIN_DEFLECTION: <number -15.0 to 15.0>
ANOMALY: <YES or NO>
REASONING: <one sentence explaining your reasoning>
STATUS: <ON_TRACK or WARNING or CRITICAL>"""


def ask_nemotron(t, roll, alt, vel, conds, phase, goal, ce, cg, recent, pid_fin=0.0):
    import re
    cond_str = ",".join(conds) if conds else "none"
    recent_str = ", ".join(recent[-3:]) if recent else "none"
    roll_trend = ""
    if len(recent) >= 2:
        try:
            last_roll = float(recent[-1].split("roll=")[0].split()[-1] if "roll=" not in recent[-1] else "0")
        except: last_roll = 0
        if abs(roll) < abs(last_roll) * 0.7: roll_trend = " (SETTLING — roll reducing)"
        elif abs(roll) > abs(last_roll) * 1.3: roll_trend = " (ESCALATING — roll increasing)"
        else: roll_trend = " (OSCILLATING — roll changing direction)"

    prompt = f"""{SYSTEM_PROMPT}

CURRENT STATE:
- Time: T+{t:.1f}s | Phase: {phase}
- Roll rate: {roll:+.1f} deg/s{roll_trend}
- Altitude: {alt:.0f}m | Conditions: {cond_str}
- Recent AI decisions: [{recent_str}]

What is your fin deflection command? Reason about the trend, not just the instant value."""

    try:
        t0 = time.time()
        r = req.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=30)
        elapsed = time.time() - t0
        text = r.json().get("response", "").strip()
        # Strip <think> blocks from reasoning models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        print(f"[AI T+{t:.0f}s] {repr(text[:100])}")
        result = {"fin": pid_fin, "anomaly": "NO", "reasoning": "nominal", "status": "ON_TRACK",
                  "tps": round(len(text) / max(elapsed, 0.1) / 4, 1)}
        for line in text.split('\n'):
            line = line.strip()
            if 'FIN_DEFLECTION' in line and ':' in line:
                try:
                    nums = re.findall(r'-?\d+\.?\d*', line.split(':',1)[1])
                    if nums: result["fin"] = max(-15, min(15, float(nums[0])))
                except: pass
            elif 'ANOMALY' in line and ':' in line:
                result["anomaly"] = line.split(':',1)[1].strip()[:20]
            elif 'REASONING' in line and ':' in line:
                result["reasoning"] = line.split(':',1)[1].strip()[:200]
            elif 'STATUS' in line and ':' in line:
                result["status"] = line.split(':',1)[1].strip()[:20]
        return result
    except Exception as e:
        print(f"[AI error] {e}")
        return {"fin": pid_fin, "anomaly": "NO", "reasoning": f"Err:{str(e)[:20]}", "status": "WARNING", "tps": 0}


def run_mission(mission):
    global state
    coast_goal = mission.get("coast",300)
    conds = mission.get("conditions",[])
    goal = mission.get("goal","Fly and return safely")
    data = load_data(coast_goal)
    state.update({
        "running":True,"status":"FLIGHT ACTIVE",
        "decisions":[],"roll_history":[],"ai_history":[],
        "pid_history":[],"time_history":[],"alt_history":[],
        "total_decisions":0,"direction_accuracy":0,
        "coast_goal":coast_goal,"coast_elapsed":0,
        "events":[],"mission_result":"","mission_goal":goal
    })
    cs=None; recent=[]; ai_all=[]; pid_all=[]
    key_times=(list(range(0,12,3))+list(range(15,min(80,coast_goal+12),10))+
               list(range(80,coast_goal+12,15))+list(range(coast_goal+12,coast_goal+100,5)))
    for (t,roll_r,pid_fin,alt_r,vel) in data:
        if not state["running"]: break
        roll,alt,vel2,active = inject(t,roll_r,alt_r,vel,conds)
        if cs is None and t>=12:
            cs=t; state["events"].append({"time":f"T+{t:.0f}s","type":"COAST_START","desc":"Entered coast phase"})
        ce=(t-cs) if cs else 0
        phase=get_phase(t,alt,vel,cs,coast_goal)
        should_call=(any(abs(t-kt)<0.6 for kt in key_times) or abs(roll)>150)
        if should_call:
            res = ask_nemotron(t,roll,alt,vel,active,phase,goal,ce,coast_goal,recent,pid_fin)
            ai_fin=res["fin"]; reasoning=res["reasoning"]; status=res["status"]
            state["tokens_per_sec"]=res["tps"]
            recent.append(f"T+{t:.0f}s {phase} fin={ai_fin:+.0f}")
            if len(recent)>5: recent.pop(0)
            if "YES" in res.get("anomaly","NO").upper():
                state["events"].append({"time":f"T+{t:.0f}s","type":"ANOMALY","desc":res.get("anomaly","")[:60]})
            state["decisions"].insert(0,{
                "time":f"T+{t:.1f}s","roll":f"{roll:+.0f}","ai_fin":f"{ai_fin:+.1f}",
                "pid_fin":f"{pid_fin:+.1f}","phase":phase,"reasoning":reasoning[:200],
                "status":status,"conditions":active[0][:35] if active else "",
                "match":"✅" if abs(ai_fin-pid_fin)<4 else "⚠️"
            })
            if len(state["decisions"])>25: state["decisions"]=state["decisions"][:25]
            state["total_decisions"]+=1
        else:
            ai_fin=pid_fin; reasoning="nominal"; status="ON_TRACK"
        ai_all.append(ai_fin); pid_all.append(pid_fin)
        state.update({
            "time":round(t,1),"roll_rate":round(roll,1),"altitude":round(alt,0),
            "velocity":round(vel,1),"ai_fin":round(ai_fin,1),"pid_fin":round(pid_fin,1),
            "phase":phase,"active_conditions":active,"coast_elapsed":round(ce,0),"reasoning":reasoning
        })
        for lst,val in [("roll_history",round(roll,1)),("ai_history",round(ai_fin,1)),
                         ("pid_history",round(pid_fin,1)),("time_history",round(t,1)),("alt_history",round(alt,0))]:
            state[lst].append(val)
            if len(state[lst])>60: state[lst]=state[lst][-60:]
        if len(ai_all)>3:
            state["direction_accuracy"]=round(float(np.mean(np.sign(np.array(ai_all))==np.sign(np.array(pid_all)))*100),1)
        time.sleep(0.2)
    ce_final=(state["time"]-cs) if cs else 0
    state["mission_result"]="✅ MISSION SUCCESS" if ce_final>=coast_goal else "✅ LANDED" if state["altitude"]<30 else "⚡ PARTIAL"
    state.update({"running":False,"status":"MISSION COMPLETE","phase":"TOUCHDOWN"})

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/state')
def get_state(): return jsonify(state)

@app.route('/api/presets')
def get_presets(): return jsonify(PRESETS)

@app.route('/api/launch',methods=['POST'])
def launch():
    if state["running"]: return jsonify({"status":"already running"})
    data=request.json or {}
    mission=PRESETS.get(data.get("preset","standard"),PRESETS["standard"]).copy()
    if data.get("custom_goal"): mission["goal"]=data["custom_goal"]
    if data.get("custom_conditions"): mission["conditions"]=data["custom_conditions"]
    if data.get("coast_duration"): mission["coast"]=int(data["coast_duration"])
    threading.Thread(target=run_mission,args=(mission,),daemon=True).start()
    return jsonify({"status":"launched","mission":mission["name"]})

@app.route('/api/abort',methods=['POST'])
def abort():
    state.update({"running":False,"status":"ABORTED","phase":"ABORTED","mission_result":"❌ ABORTED"})
    return jsonify({"status":"aborted"})

@app.route('/api/reset',methods=['POST'])
def reset():
    state.update({
        "running":False,"status":"STANDBY","phase":"READY","time":0,"roll_rate":0,
        "altitude":0,"velocity":0,"ai_fin":0,"pid_fin":0,"action":"STANDBY",
        "reasoning":"Configure your mission and hit LAUNCH",
        "decisions":[],"roll_history":[],"ai_history":[],"pid_history":[],
        "time_history":[],"alt_history":[],"total_decisions":0,
        "direction_accuracy":0,"coast_elapsed":0,"active_conditions":[],
        "events":[],"mission_result":"","tokens_per_sec":0
    })
    return jsonify({"status":"reset"})

if __name__=='__main__':
    print("\n🚀 AutoClaw Mission Control")
    print(f"   Model: {MODEL}")
    print("   Open http://0.0.0.0:5000\n")
    app.run(host='0.0.0.0',port=5000,debug=False)