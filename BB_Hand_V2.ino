/* ───────────────────  B B  _ H A N D  _ V 2  ────────────────────
   • 4 analogue-feedback PWM servos → ab/ad (Index…Pinky)
   • 5 SCServo servos              → flex  (Thumb…Pinky)
     Thumb keeps ID-0  → broadcast only, no feedback
   • Ring ↔ Pinky wiring is swapped → FLEX_MAP fixes it
   • UART0 1 000 000 baud   TX=GP12   RX=GP13  (Philhower core)   or
     UART0 default pins   GP0 / GP1  (mbed core, no pin remap)
   ---------------------------------------------------------------- */

#include <Servo.h>
#include <SCServo.h>

#define NUM_PWM 4
#define NUM_SCS 5                                       // thumb … pinky

/* ── pins & IDs ────────────────────────────────────────────── */
const uint8_t PWM_PIN[NUM_PWM] = {6, 7, 8, 9};
const uint8_t FB_PIN [NUM_PWM] = {26,27,28,29};

const uint8_t SCS_ID[NUM_SCS]  = {1, 2, 3, 4, 5};

/* Thumb,Index,Middle,Ring,Pinky  →  servo-index in SCS_ID[]      *
 * Ring & Pinky are swapped, so map 3↔4                            */
const uint8_t FLEX_MAP[NUM_SCS] = {0, 1, 2, 3, 4};

/* ── ranges & directions ───────────────────────────────────── */
const int  PWM_MIN_REL[NUM_PWM] = {-90,-90,-90,-90};
const int  PWM_MAX_REL[NUM_PWM] = { 90, 90, 90, 90};

int8_t PWM_DIR[NUM_PWM] = { 1,  1,  1,  1};
int8_t SCS_DIR[NUM_SCS] = {1, -1, -1, -1, 1};   // pinky forward (+1)

int8_t PWM_NEUTRAL_MODE[NUM_PWM] = { 1, 0, 0, -1};   // idx upper, pinky lower

const int SCS_MIN_DEG[NUM_SCS] = {20, 20, 20, 20, 20};
const int SCS_MAX_DEG[NUM_SCS] = {270,270,270,270,270};

/* ── globals ───────────────────────────────────────────────── */
Servo  pwm[NUM_PWM];
SCSCL  scs;
String buf;

int  adcMin[NUM_PWM], adcMax[NUM_PWM];
int  degMin[NUM_PWM], degMax[NUM_PWM];
int  pwmNeutralAbs[NUM_PWM], pwmRelMin[NUM_PWM], pwmRelMax[NUM_PWM];
float pwmScale[NUM_PWM];
int  lastRel[NUM_PWM] = {0};

/* ── helpers ───────────────────────────────────────────────── */
static inline float mapF(long x,long inMin,long inMax,long outMin,long outMax){
  return (float)(x-inMin)*(outMax-outMin)/(float)(inMax-inMin)+outMin;
}
static inline uint16_t deg2cnt(int d){ return uint16_t(d*1000UL/270UL); }

/* ── calibration ──────────────────────────────────────────── */
void calibratePWM(uint8_t i){
  int absMin=90+PWM_MIN_REL[i], absMax=90+PWM_MAX_REL[i], rng=absMax-absMin;
  degMin[i]=absMin; degMax[i]=absMax;

  pwm[i].write(absMin); delay(800); adcMin[i]=analogRead(FB_PIN[i]);
  pwm[i].write(absMax); delay(800); adcMax[i]=analogRead(FB_PIN[i]);

  if(PWM_NEUTRAL_MODE[i]==-1){
    pwmNeutralAbs[i]=absMin; pwmRelMin[i]=0;  pwmRelMax[i]=30; pwmScale[i]=rng/30.0f;
  }else if(PWM_NEUTRAL_MODE[i]==1){
    pwmNeutralAbs[i]=absMax; pwmRelMin[i]=-30; pwmRelMax[i]=0; pwmScale[i]=rng/30.0f;
  }else{
    pwmNeutralAbs[i]=(absMin+absMax)/2; pwmRelMin[i]=-30; pwmRelMax[i]=30; pwmScale[i]=rng/60.0f;
  }
  pwm[i].write(pwmNeutralAbs[i]); delay(600);
}

/* ── setup ─────────────────────────────────────────────────── */
void setup() {
  Serial.begin(1000000);                          // fast USB debug

#if defined(ARDUINO_ARCH_RP2040) && !defined(ARDUINO_ARCH_MBED)
  /* Philhower “Arduino-Pico” core → pin remap is available */
  Serial1.setTX(12); Serial1.setRX(13);
#endif
  Serial1.begin(1000000);                         // UART0 1 Mb s
  scs.pSerial = &Serial1;

  for (uint8_t i = 0; i < NUM_PWM; ++i) {
    pwm[i].attach(PWM_PIN[i]);
    calibratePWM(i);
  }
  Serial.println("READY");
}

/* ── motion ───────────────────────────────────────────────── */
void movePWM(uint8_t idx,int rel){
  int req=rel; rel=constrain(rel,pwmRelMin[idx],pwmRelMax[idx]);
  int absDeg=pwmNeutralAbs[idx]+rel*pwmScale[idx]*PWM_DIR[idx]+0.5f;
  absDeg=constrain(absDeg,degMin[idx],degMax[idx]); pwm[idx].write(absDeg);
  lastRel[idx]=rel; if(req!=rel){ Serial.print("PWM "); Serial.print(idx+1); Serial.println(" clipped"); }
}
void moveSCS(uint8_t sIdx,int deg){
  uint8_t id=SCS_ID[sIdx];
  int req=deg; if(SCS_DIR[sIdx]<0) deg=270-deg;
  deg=constrain(deg,SCS_MIN_DEG[sIdx],SCS_MAX_DEG[sIdx]);

  if(id==0){                                       // broadcast – thumb only
      Serial.println("WARN: Thumb is ID-0 (broadcast only, no feedback)");
      scs.WritePos(0, deg2cnt(deg), 0, 0);         // moves *all* servos
  }else{
      scs.WritePos(id, deg2cnt(deg), 0, 0);
  }
  if(req!=deg){ Serial.print("SCS "); Serial.print(sIdx); Serial.println(" clipped"); }
}

/* ── burst parser (A …) ───────────────────────────────────── */
bool parseAllCmd(const String &line){
  const uint8_t need=NUM_PWM+NUM_SCS; int32_t v[need];
  uint8_t n=0; int pos=2;
  while(n<need){
    int nxt=line.indexOf(' ',pos);
    String tok=(nxt==-1)?line.substring(pos):line.substring(pos,nxt);
    if(!tok.length()) break; v[n++]=tok.toInt(); if(nxt==-1) break; pos=nxt+1;
  }
  if(n!=need) return false;

  for(uint8_t i=0;i<NUM_PWM;i++) movePWM(i,v[i]);          // ab/ad
  for(uint8_t f=0;f<NUM_SCS;f++) moveSCS(FLEX_MAP[f],v[NUM_PWM+f]); // flex
  return true;
}

/* ── utilities ───────────────────────────────────────────── */
void torqueOff(){ for(uint8_t i=0;i<NUM_SCS;i++) scs.EnableTorque(SCS_ID[i],0); }
void snapshot(){
  Serial.println("=== ANGLES ===");
  for(uint8_t i=0;i<NUM_PWM;i++){
    int a=analogRead(FB_PIN[i]);
    float d=mapF(a,adcMin[i],adcMax[i],degMin[i],degMax[i]);
    Serial.print("PWM "); Serial.print(i+1); Serial.print(" cmd ");
    Serial.print(lastRel[i]); Serial.print(" fb "); Serial.println(d,1);
  }
  for(uint8_t i=0;i<NUM_SCS;i++){
    uint8_t id=SCS_ID[i];
    if(id==0){ Serial.print("SCS "); Serial.print(i); Serial.println(" : N/A"); continue; }
    int16_t p=scs.ReadPos(id);
    float d=(p<0)?NAN:p*270.0/1000.0;
    Serial.print("SCS "); Serial.print(i); Serial.print(" : ");
    if(isnan(d)) Serial.println("ERR"); else Serial.println(d,1);
  }
}

/* ── command line ────────────────────────────────────────── */
void exec(String s){
  s.trim(); s.toUpperCase();
  if(s=="E"){ torqueOff(); return; }
  if(s=="R"){ snapshot();  return; }

  char t=s.charAt(0);
  if(t=='A'){ if(!parseAllCmd(s)) Serial.println("ERR A fmt"); return; }

  int sp1=s.indexOf(' '), sp2=s.indexOf(' ',sp1+1);
  if(sp1==-1||sp2==-1){ Serial.println("ERR fmt"); return; }
  uint8_t idx=s.substring(sp1+1,sp2).toInt(); int val=s.substring(sp2+1).toInt();

  if(t=='P')      movePWM(idx-1,val);
  else if(t=='S'){ if(idx<NUM_SCS) moveSCS(idx,val); else Serial.println("ERR idx"); }
  else            Serial.println("ERR cmd");
}

void loop(){
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n'||c=='\r'){ if(buf.length()) exec(buf); buf=""; }
    else buf+=c;
  }
}
