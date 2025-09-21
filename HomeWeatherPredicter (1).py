# Enhanced "HomeForecast" with cooling & heating planning, forecast, enthalpy, solar bias, and stability improvements
import appdaemon.plugins.hass.hassapi as hass
from datetime import datetime, timezone, timedelta
import math
import requests
import json


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# --- Psychrometrics (safe) ---
ATM_PA = 101325.0


def _f_to_c(f):
    try:
        f = float(f)
    except (ValueError, TypeError):
        f = 68.0  # 20C
    return (f - 32.0) * (5.0 / 9.0)


def _psat_pa_tc(tc):
    tc = max(-50.0, min(60.0, float(tc)))
    return 610.94 * math.exp(17.625 * tc / (tc + 243.04))


def _enthalpy_kj_per_kg_dryair(temp_f, rh):
    tc = _f_to_c(temp_f)
    tc = max(-50.0, min(60.0, tc))
    try:
        rhf = float(rh)
    except (ValueError, TypeError):
        rhf = 50.0
    rhf = max(1.0, min(99.0, rhf)) / 100.0

    ps = _psat_pa_tc(tc)
    pv = rhf * ps
    denom = max(100.0, (ATM_PA - pv))
    w = 0.62198 * pv / denom
    return 1.006 * tc + w * (2501.0 + 1.86 * tc)


class HomeWeatherPredicter(hass.Hass):
    """
    dTin/dt = a*(Tout - Tin) + kH*Ih + kC*Ic + b + kE*(hout - hin) + kS*Solar
    theta = [a, kH, kC, b, kE, kS]
    """
    # --- Entity Configuration ---
    H_INDOOR_T = "input_text.home_model_indoor_sensor"
    H_OUTDOOR_T = "input_text.home_model_outdoor_sensor"
    H_CLIMATE = "input_text.home_model_climate_entity"
    H_INDOOR_RH = "input_text.home_model_indoor_humidity_entity"
    H_OUTDOOR_RH = "input_text.home_model_outdoor_humidity_entity"
    H_ENABLE = "input_boolean.home_model_learning_enabled"
    H_TAU = "input_number.home_model_tau_hours"
    H_KH = "input_number.home_model_k_heat"
    H_KC = "input_number.home_model_k_cool"
    H_B = "input_number.home_model_bias"
    H_LAMBDA = "input_number.home_model_forgetting_factor"
    H_PERIOD = "input_number.home_model_update_minutes"
    H_CAPF = "input_number.home_model_comfort_cap_f"
    H_FLOORF = "input_number.home_model_heat_min_f"
    H_HRS = "input_number.home_model_forecast_hours"
    H_RECO_DELAY = "input_number.home_model_recommendation_cooldown"
    H_AW_TOKEN = "input_text.accuweather_token"
    H_AW_KEY = "input_text.accuweather_location_key"
    H_STORAGE = "input_text.home_model_storage"

    # --- Published Sensors ---
    S_TAU = "sensor.home_model_tau_hours"
    S_KH = "sensor.home_model_k_heat"
    S_KC = "sensor.home_model_k_cool"
    S_B = "sensor.home_model_bias"
    S_KE = "sensor.home_model_k_enthalpy"
    S_KS = "sensor.home_model_k_solar"
    S_LOSS = "sensor.home_model_fit_loss"
    S_SAM = "sensor.home_model_samples"
    S_FC_SER = "sensor.home_model_indoor_forecast_12h"
    S_OUT_SER = "sensor.home_model_outdoor_forecast_12h"
    S_F30 = "sensor.home_temp_plus_30m_model"
    S_F60 = "sensor.home_temp_plus_60m_model"
    S_F180 = "sensor.home_temp_plus_180m_model"
    S_T2CAP = "sensor.home_model_time_to_cap_minutes"
    S_T2FLR = "sensor.home_model_time_to_floor_minutes"
    S_IDEAL_COOL = "sensor.home_model_ideal_cool_start_time"
    S_IDEAL_HEAT = "sensor.home_model_ideal_heat_start_time"
    S_COOL_OFF = "sensor.home_model_cool_off_time"
    S_HEAT_OFF = "sensor.home_model_heat_off_time"
    S_RECO = "sensor.home_model_control_recommendation"
    S_INDOOR_ACC = "sensor.home_model_indoor_accuracy"
    S_OUTDOOR_ACC = "sensor.home_model_outdoor_accuracy"

    # --- Parameter Limits ---
    MIN_TAU_H, MAX_TAU_H = 0.5, 72.0
    MIN_KH, MAX_KH = 0.0, 2.0
    MIN_KC, MAX_KC = -2.0, 0.0
    MIN_B, MAX_B = -0.2, 0.2
    MIN_KE, MAX_KE = -0.02, 0.02
    MIN_KS, MAX_KS = -0.002, 0.002

    def initialize(self):
        self.theta, self.P, self.samples = None, None, 0
        self.last_tin, self.last_ts = None, None
        self._aw_cache, self._aw_cache_ts = None, None
        self._last_reco_ts, self._pending_reco = None, None
        self._past_preds = [] # For indoor accuracy
        self._outdoor_errors = [] # For outdoor accuracy

        self.run_interval_seconds = int(self.args.get("run_interval_seconds", 60))
        self._publish_placeholders()
        self._load_model_state()

        a, kH, kC, b, kE, kS = self.theta
        self._publish_params(a, kH, kC, b, kE, kS, loss=None)
        self.set_state(self.S_SAM, state=self.samples, attributes={"friendly_name": "Home Model samples"})

        # --- Schedulers and Listeners ---
        self.run_every(self._tick, self.datetime() + timedelta(seconds=5), self.run_interval_seconds)
        self.run_every(self._refresh_aw_fcst, self.datetime() + timedelta(seconds=10), 30 * 60)
        self.run_on_shutdown(self._save_model_state)

        # Listen for immediate changes to HVAC state
        climate_id_entity = self._get_text(self.H_CLIMATE, "climate.downstairs")
        self.listen_state(self._hvac_state_change, climate_id_entity, attribute="hvac_action")

        self.log("HomeWeatherPredicter (enhanced) initialized", level="INFO")

    def _hvac_state_change(self, entity, attribute, old, new, kwargs):
        self.log(f"HVAC action changed from '{old}' to '{new}'. Triggering immediate tick.", level="DEBUG")
        self.run_in(self._tick, 1) # Run a tick in 1 second to react

    def _publish_placeholders(self):
        # Initialize all sensors to a known state on startup
        entities = [
            self.S_F30, self.S_F60, self.S_F180, self.S_IDEAL_COOL, self.S_IDEAL_HEAT,
            self.S_COOL_OFF, self.S_HEAT_OFF, self.S_RECO, self.S_T2CAP, self.S_T2FLR,
            self.S_FC_SER, self.S_OUT_SER, self.S_INDOOR_ACC, self.S_OUTDOOR_ACC
        ]
        for e in entities:
            attrs = {}
            if "temp" in e or "forecast" in e:
                attrs = {"unit_of_measurement": "°F", "device_class": "temperature", "state_class": "measurement"}
            if "accuracy" in e:
                attrs = {"unit_of_measurement": "°F", "state_class": "measurement", "friendly_name": e.split('.')[-1].replace('_', ' ').title()}
            self.set_state(e, state="unknown", attributes=attrs)

    def _theta_from_helpers(self):
        tau_h = self._get_float(self.H_TAU, 12.0)
        a = 1.0 / max(1e-6, tau_h * 60.0)
        kH = self._get_float(self.H_KH, 0.30)
        kC = self._get_float(self.H_KC, -0.25)
        b = self._get_float(self.H_B, 0.0)
        kE = 0.0 # Will be learned
        kS = 0.0 # Will be learned
        return [a, kH, kC, b, kE, kS]

    def _get_text(self, entity, default=""):
        v = self.get_state(entity)
        return v.strip() if isinstance(v, str) else default

    def _get_float(self, entity, default=None):
        v = self.get_state(entity)
        try:
            f = float(v)
            return f if not math.isnan(f) else default
        except (ValueError, TypeError):
            return default

    def _hvac_action(self, climate_id):
        try:
            attrs = self.get_state(climate_id, attribute="all")["attributes"]
            act = attrs.get("hvac_action")
            return act if act in ("heating", "cooling") else "idle"
        except Exception:
            return "idle"

    # --- State Persistence ---
    def _load_model_state(self):
        try:
            stored_state = self.get_state(self.H_STORAGE)
            if stored_state and stored_state != "unknown":
                data = json.loads(stored_state)
                self.theta = data['theta']
                self.P = data['P']
                self.samples = data['samples']
                self.log("Successfully loaded model state from storage.", level="INFO")
            else:
                raise ValueError("No stored state found")
        except Exception as e:
            self.log(f"Could not load model state, initializing from helpers: {e}", level="WARNING")
            self.theta = self._theta_from_helpers()
            self.P = [[1000.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
            self.samples = 0

    def _save_model_state(self, kwargs=None):
        try:
            state_to_save = {
                'theta': self.theta,
                'P': self.P,
                'samples': self.samples
            }
            self.call_service("input_text/set_value", entity_id=self.H_STORAGE, value=json.dumps(state_to_save))
            self.log("Successfully saved model state to storage.", level="INFO")
        except Exception as e:
            self.log(f"Error saving model state: {e}", level="ERROR")

    # --- AccuWeather Fetch (Unchanged from original) ---
    def _refresh_aw_fcst(self, kwargs):
        token = self._get_text(self.H_AW_TOKEN, "")
        loc = self._get_text(self.H_AW_KEY, "")
        if not token or not loc:
            # fine to run without AW; we’ll fall back to current outdoor values
            self.log("AccuWeather disabled (missing token/key).", level="DEBUG")
            return

        url = f"https://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{loc}?details=true"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                raw = r.json() or []
                slim = []
                for it in raw:
                    dt = it.get("DateTime")
                    tF = (it.get("Temperature") or {}).get("Value")
                    rh = it.get("RelativeHumidity")
                    sol = (it.get("SolarIrradiance") or {}).get("Value", 0.0) or 0.0
                    if isinstance(tF, (int, float)) and dt:
                        slim.append({"DateTime": dt, "TempF": float(tF), "RH": rh, "Solar": float(sol)})

                if not slim:
                    self.log("AccuWeather: empty/invalid 12h payload.", level="WARNING")
                    return

                self._aw_cache = slim
                fetched_at = datetime.now(timezone.utc)
                self._aw_cache_ts = fetched_at

                temps = [x["TempF"] for x in slim]
                avg = round(sum(temps) / len(temps), 1)

                # summary
                self.set_state(
                    self.S_EXT,
                    state=avg,
                    attributes={
                        "count": len(slim),
                        "first": slim[0]["DateTime"],
                        "last": slim[-1]["DateTime"],
                        "fetched_at": fetched_at.astimezone().isoformat(timespec="seconds"),
                        "source": "accuweather",
                        "location_key": loc,
                        "temps": [round(v, 1) for v in temps[:12]],
                    },
                )

                # outdoor trajectory (12h)
                try:
                    times = [it["DateTime"] for it in slim]
                    series = [{"t": it["DateTime"], "y": round(it["TempF"], 1)} for it in slim]
                    state_now = series[0]["y"] if series else "unknown"
                    self.set_state(
                        self.S_OUT_SER,
                        state=state_now,
                        attributes={
                            "unit_of_measurement": "°F",
                            "device_class": "temperature",
                            "state_class": "measurement",
                            "friendly_name": "Outdoor Forecast (12h)",
                            "times": times,
                            "temps": [pt["y"] for pt in series],
                            "series": series,
                            "count": len(series),
                            "location_key": loc,
                        },
                    )
                except Exception as pe:
                    self.log(f"AccuWeather: failed to publish outdoor trajectory: {pe}", level="WARNING")

            elif r.status_code in (401, 403):
                self.log(f"AccuWeather auth failed ({r.status_code}). Check token/permissions.", level="ERROR")
            elif r.status_code == 429:
                hdr = r.headers.get("RateLimit-Reset") or r.headers.get("Retry-After")
                try:
                    delay = int(hdr)
                except Exception:
                    delay = 120
                self.log(f"AccuWeather rate-limited (429). Retrying in {delay}s.", level="WARNING")
                self.run_in(self._refresh_aw_fcst, delay)
            else:
                self.log(f"AccuWeather HTTP {r.status_code}: {r.text[:160]}", level="WARNING")

        except Exception as e:
            self.log(f"AccuWeather error: {e}", level="WARNING")

    # --- Model & Simulation ---
    def _rls_update(self, theta, P, x, y, lam):
        n = 6
        v = [sum(P[i][j] * x[j] for j in range(n)) for i in range(n)]
        xTPx = sum(x[i] * v[i] for i in range(n))
        denom = lam + xTPx
        if abs(denom) < 1e-9: return theta, P, 0 # Avoid division by zero
        K = [vi / denom for vi in v]
        yhat = sum(theta[i] * x[i] for i in range(n))
        err = y - yhat
        theta_new = [theta[i] + K[i] * err for i in range(n)]
        KvT = [[K[i] * v[j] for j in range(n)] for i in range(n)]
        P_new = [[(P[i][j] - KvT[i][j]) / lam for j in range(n)] for i in range(n)]
        return theta_new, P_new, err

    def _simulate(self, Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes, action_fn, theta):
        a, kH, kC, b, kE, kS = theta
        T = float(Tin)
        traj = []
        for m in range(1, minutes + 1):
            i = (m - 1) // 60
            Tout = float(Tout_series[i] if i < len(Tout_series) else Tout_series[-1])
            RHin = RH_in_series[i] if i < len(RH_in_series) else RH_in_series[-1]
            RHout = RH_out_series[i] if i < len(RH_out_series) else RH_out_series[-1]
            Solar = float(Solar_series[i] if i < len(Solar_series) else Solar_series[-1])

            act = action_fn(m)
            Ih = 1.0 if act == "heating" else 0.0
            Ic = 1.0 if act == "cooling" else 0.0

            hin = _enthalpy_kj_per_kg_dryair(T, RHin or 50.0)
            hout = _enthalpy_kj_per_kg_dryair(Tout, RHout or 50.0)

            dTdt = a * (Tout - T) + kH * Ih + kC * Ic + b + kE * (hout - hin) + kS * Solar
            dTdt = max(-2.0, min(2.0, dTdt))
            T += dTdt
            T = max(-40.0, min(140.0, T))
            traj.append(T)
        return traj

    def _build_drivers(self, horizon_h, Tout_now, RHout_now, RHin_now):
        fc = self._aw_cache or []
        H = max(1, min(horizon_h, len(fc) if fc else 1))
        Tout_series, RH_out_series, Solar_series, RH_in_series = [], [], [], []
        for h in range(H):
            if fc and h < len(fc):
                tF = fc[h]["TempF"] if isinstance(fc[h].get("TempF"), (int, float)) else Tout_now
                rh = fc[h]["RH"] if isinstance(fc[h].get("RH"), (int, float)) else (RHout_now if RHout_now is not None else 50.0)
                sol = fc[h].get("Solar", 0.0) or 0.0
            else:
                tF, rh, sol = Tout_now, (RHout_now if RHout_now is not None else 50.0), 0.0
            if h == 0:
                tF = 0.7 * Tout_now + 0.3 * tF
                rh = 0.7 * (RHout_now if RHout_now is not None else 50.0) + 0.3 * rh
            Tout_series.append(tF)
            RH_out_series.append(rh)
            Solar_series.append(sol)
            RH_in_series.append(RHin_now if RHin_now is not None else 50.0)

        if not Tout_series:
            Tout_series = [Tout_now]
            RH_out_series = [RHout_now if RHout_now is not None else 50.0]
            Solar_series = [0.0]
            RH_in_series = [RHin_now if RHin_now is not None else 50.0]
            H = 1

    # --- Main Tick ---
    def _tick(self, kwargs=None):
        try:
            # --- Get Config and Sensor Values ---
            lam = float(self._get_float(self.H_LAMBDA, 0.995))
            learning_enabled = (self.get_state(self.H_ENABLE) == "on")
            capF = float(self._get_float(self.H_CAPF, 80.0))
            floorF = float(self._get_float(self.H_FLOORF, 62.0))
            horizon_h = int(self._get_float(self.H_HRS, 12))
            reco_delay_min = int(self._get_float(self.H_RECO_DELAY, 15))

            indoor_id = self._get_text(self.H_INDOOR_T, "sensor.indoor_temperature")
            outdoor_id = self._get_text(self.H_OUTDOOR_T, "sensor.accuweather_current_temperature")
            climate_id = self._get_text(self.H_CLIMATE, "climate.downstairs")
            in_rh_id = self._get_text(self.H_INDOOR_RH, "")
            out_rh_id = self._get_text(self.H_OUTDOOR_RH, "")

            Tin = self._get_float(indoor_id, None)
            Tout_now = self._get_float(outdoor_id, None)
            RHin_now = self._get_float(in_rh_id, None)
            RHout_now = self._get_float(out_rh_id, None)
            if Tin is None or Tout_now is None: return

            now = datetime.now(timezone.utc)
            if self.last_ts is None or self.last_tin is None:
                self.last_ts, self.last_tin = now, Tin
                return

            dt_min = (now - self.last_ts).total_seconds() / 60.0
            if dt_min < 0.9: return # Avoid rapid-fire updates

            # --- Accuracy Calculation ---
            # Indoor: Compare current temp with prediction made `dt_min` ago
            if self._past_preds:
                pred_for_now = self._past_preds.pop(0)
                error = abs(Tin - pred_for_now)
                self.set_state(self.S_INDOOR_ACC, state=round(error, 2))

            # Outdoor: Compare current outdoor temp with forecast
            if self._aw_cache:
                fc_temp_now = self._aw_cache[0]['TempF']
                outdoor_error = abs(Tout_now - fc_temp_now)
                self._outdoor_errors.append(outdoor_error)
                if len(self._outdoor_errors) > 60: self._outdoor_errors.pop(0) # 1hr rolling avg
                avg_outdoor_error = sum(self._outdoor_errors) / len(self._outdoor_errors)
                self.set_state(self.S_OUTDOOR_ACC, state=round(avg_outdoor_error, 2))


            # --- Learning Update ---
            act_now = self._hvac_action(climate_id)
            solar_now = (self._aw_cache[0]['Solar'] if self._aw_cache and 'Solar' in self._aw_cache[0] else 0.0) or 0.0
            hin = _enthalpy_kj_per_kg_dryair(Tin, RHin_now or 50.0)
            hout = _enthalpy_kj_per_kg_dryair(Tout_now, RHout_now or 50.0)
            y = (Tin - self.last_tin) / dt_min
            x = [
                (Tout_now - Tin),
                1.0 if act_now == "heating" else 0.0,
                1.0 if act_now == "cooling" else 0.0,
                1.0,
                (hout - hin),
                solar_now,
            ]
            
            err = None
            if learning_enabled:
                self.theta, self.P, err = self._rls_update(self.theta, self.P, x, y, lam)
                # Apply constraints
                self.theta[0] = clip(self.theta[0], 1.0 / (self.MAX_TAU_H * 60.0), 1.0 / (self.MIN_TAU_H * 60.0))
                self.theta[1] = clip(self.theta[1], self.MIN_KH, self.MAX_KH)
                self.theta[2] = clip(self.theta[2], self.MIN_KC, self.MAX_KC)
                self.theta[3] = clip(self.theta[3], self.MIN_B, self.MAX_B)
                self.theta[4] = clip(self.theta[4], self.MIN_KE, self.MAX_KE)
                self.theta[5] = clip(self.theta[5], self.MIN_KS, self.MAX_KS)
                self.samples += 1

            a, kH, kC, b, kE, kS = self.theta
            loss = err * err if err is not None else None
            self._publish_params(a, kH, kC, b, kE, kS, loss)
            self.set_state(self.S_SAM, state=self.samples)

            # --- Drivers & Simulation ---
            H, Tout_series, RH_out_series, Solar_series, RH_in_series = self._build_drivers(horizon_h, Tout_now, RHout_now, RHin_now)
            if not H: return # Not enough data to simulate

            # --- Idle Forecast & Recommendation ---
            traj_idle = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, H * 60, lambda m: "idle", self.theta)
            self._past_preds.append(traj_idle[0]) # Store prediction for next tick's accuracy check
            if len(self._past_preds) > 5: self._past_preds.pop(0)

            max_idle = max(traj_idle) if traj_idle else Tin
            min_idle = min(traj_idle) if traj_idle else Tin
            
            # --- Recommendation with Hysteresis ---
            current_reco = "none"
            if max_idle > capF:
                current_reco = "cool"
            elif min_idle < floorF:
                current_reco = "heat"

            last_reco = self.get_state(self.S_RECO)
            if current_reco != last_reco:
                if self._pending_reco and self._pending_reco['reco'] == current_reco:
                    if (now - self._pending_reco['ts']).total_seconds() / 60 >= reco_delay_min:
                        self.set_state(self.S_RECO, state=current_reco)
                        self._pending_reco = None
                else:
                    self._pending_reco = {'reco': current_reco, 'ts': now}
            else:
                self._pending_reco = None # Reset if reco matches current state
            
            final_reco = self.get_state(self.S_RECO) # Use the stable, committed recommendation

            # --- Ideal start times (latest acceptable start) ---
            latest_cool = None
            if final_reco == "cool":
                def action_cool_from(start_m):
                    return lambda m: "cooling" if m >= start_m else "idle"
                # Iterate backwards from the end of the horizon to find the latest possible start time
                for start_m in range(H * 60, 0, -1):
                    trajc = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_cool_from(start_m), theta=self.theta)
                    if all(v <= capF + 0.01 for v in trajc):
                        latest_cool = start_m
                        break
                if latest_cool is None: # If no suitable start time was found, it means we must start now.
                    latest_cool = 0

            latest_heat = None
            if final_reco == "heat":
                def action_heat_from(start_m):
                    return lambda m: "heating" if m >= start_m else "idle"
                # Iterate backwards to find latest start time
                for start_m in range(H * 60, 0, -1):
                    trajh = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_heat_from(start_m), theta=self.theta)
                    if all(v >= floorF - 0.01 for v in trajh):
                        latest_heat = start_m
                        break
                if latest_heat is None: # Must start now
                    latest_heat = 0

            # --- Off-times (earliest stop) if currently cooling/heating ---
            cool_off_min = None
            if act_now == "cooling":
                def action_cool_until(stop_m):
                    return lambda m: "cooling" if m < stop_m else "idle"
                # Iterate forwards to find earliest stop time
                for stop_m in range(1, H * 60 + 1):
                    trajc2 = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_cool_until(stop_m), theta=self.theta)
                    if all(v <= capF + 0.01 for v in trajc2):
                        cool_off_min = stop_m
                        break

            heat_off_min = None
            if act_now == "heating":
                def action_heat_until(stop_m):
                    return lambda m: "heating" if m < stop_m else "idle"
                # Iterate forwards to find earliest stop time
                for stop_m in range(1, H * 60 + 1):
                    trajh2 = self._simulate(Tin, Tout_series, RH_in_series, RH_out_series, Solar_series, minutes=H * 60, action_fn=action_heat_until(stop_m), theta=self.theta)
                    if all(v >= floorF - 0.01 for v in trajh2):
                        heat_off_min = stop_m
                        break

            # --- Publish human-friendly start/stop times with status ---
            def mins_to_iso(m):
                return (now + timedelta(minutes=m)).astimezone().isoformat(timespec="minutes")

            # Publish Ideal Cool Start Time
            if latest_cool is not None:
                start_time_iso = mins_to_iso(latest_cool)
                status = "scheduled"
                if latest_cool <= 0:
                    status = "active" if act_now == "cooling" else "missed"
                
                self.set_state(
                    self.S_IDEAL_COOL,
                    state=start_time_iso,
                    attributes={"minutes_from_now": latest_cool, "status": status, "cap_f": capF, "horizon_hours": H}
                )
            else:
                self.set_state(self.S_IDEAL_COOL, state="not needed", attributes={"status": "not_needed"})

            # Publish Ideal Heat Start Time
            if latest_heat is not None:
                start_time_iso = mins_to_iso(latest_heat)
                status = "scheduled"
                if latest_heat <= 0:
                    status = "active" if act_now == "heating" else "missed"

                self.set_state(
                    self.S_IDEAL_HEAT,
                    state=start_time_iso,
                    attributes={"minutes_from_now": latest_heat, "status": status, "floor_f": floorF, "horizon_hours": H}
                )
            else:
                self.set_state(self.S_IDEAL_HEAT, state="not needed", attributes={"status": "not_needed"})

            # Publish Cool Off Time
            if act_now == "cooling":
                off_time_state = mins_to_iso(cool_off_min) if cool_off_min is not None else "keep running"
                self.set_state(
                    self.S_COOL_OFF,
                    state=off_time_state,
                    attributes={"minutes_from_now": cool_off_min, "cap_f": capF, "horizon_hours": H}
                )
            else:
                self.set_state(self.S_COOL_OFF, state="n/a")

            # Publish Heat Off Time
            if act_now == "heating":
                off_time_state = mins_to_iso(heat_off_min) if heat_off_min is not None else "keep running"
                self.set_state(
                    self.S_HEAT_OFF,
                    state=off_time_state,
                    attributes={"minutes_from_now": heat_off_min, "floor_f": floorF, "horizon_hours": H}
                )
            else:
                self.set_state(self.S_HEAT_OFF, state="n/a")

            # --- Publish series entity (indoor) ---
            self._publish_series(
                now, H, Tin, traj_idle, final_reco, latest_cool, latest_heat,
                Tout_series, RH_in_series, RH_out_series, Solar_series,
                capF, floorF
            )

            # --- Advance & Log ---
            self.last_tin, self.last_ts = Tin, now
            self.log(f"Tick: Tin={Tin:.1f}, Tout={Tout_now:.1f}, Action={act_now}, Solar={solar_now:.1f}", level="DEBUG")

        except Exception as e:
            self.log(f"HomeWeatherPredicter tick error: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")

    # --- Helper functions to be copied from the original script ---
    def _publish_params(self, a, kH, kC, b, kE, kS, loss=None):
        tau_h = 1.0 / (a * 60.0) if a > 0 else self.MAX_TAU_H
        self.set_state(self.S_TAU, state=round(tau_h, 3), attributes={"unit_of_measurement": "h"})
        self.set_state(self.S_KH, state=round(kH, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_KC, state=round(kC, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_B, state=round(b, 4), attributes={"unit_of_measurement": "°F/min"})
        self.set_state(self.S_KE, state=round(kE, 6), attributes={"unit_of_measurement": "°F/min per kJ/kg"})
        self.set_state(self.S_KS, state=round(kS, 6), attributes={"unit_of_measurement": "°F/min per W/m²"})
        if loss is not None:
            self.set_state(self.S_LOSS, state=round(loss, 6))

    def _publish_series(self, now, H, Tin, traj_idle, reco, latest_cool, latest_heat,
                        Tout_series, RH_in_series, RH_out_series, Solar_series,
                        capF, floorF):
        # Convert minute trajectory -> hourly points
        def minutes_to_hourly(traj_mins, hours):
            out = []
            for hh in range(1, hours + 1):
                idx = min(hh * 60 - 1, len(traj_mins) - 1)
                out.append(traj_mins[idx])
            return out
