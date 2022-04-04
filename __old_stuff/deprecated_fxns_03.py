def identify_gaps(lklc, transit_mask, break_tolerance, jump_tolerance=5.0):
    """
    Search a lk.LightCurve for large time breaks and flux jumps
    
    Parameters
    ----------
        lklc : lk.LightCurve() object
            must have time, flux, and cadenceno attributes
        transit_mask : array (bool)
            True for each cadence near transit
        break_tolerance : int
            number of cadences considered a large gap in time
        jump_tolerance : float
            number of sigma from median flux[i+1]-flux[i] to be considered a large jump in flux (default=5.0)
            
    Returns
    -------
        gap_locs : array
            indexes of identified gaps, including endpoints
    """
    
    # identify time gaps
    breaks = lklc.cadenceno[1:]-lklc.cadenceno[:-1]
    breaks = np.pad(breaks, (1,0), 'constant', constant_values=(1,0))
    break_locs = np.where(breaks > break_tolerance)[0]
    break_locs = np.pad(break_locs, (1,1), 'constant', constant_values=(0,len(breaks)+1))
    
    # identify flux jumps
    jumps = lklc.flux[1:]-lklc.flux[:-1]
    jumps = np.pad(jumps, (1,0), 'constant', constant_values=(0,0))
    big_jump = np.abs(jumps - np.median(jumps))/astropy.stats.mad_std(jumps) > 5.0
    jump_locs = np.where(~transit_mask*big_jump)[0]
    
    return np.sort(np.unique(np.hstack([break_locs, jump_locs])))


def flatten(lklc, transit_win_length, savgol_win_length, transit_mask, combo_mask,
            break_tolerance, polyorder=2, return_trend=False):
    """
    Remove long-term (low frequency) trends from a lightcurve
    Flattening applies a modified Savitsky-Golay filter that interpolates over known transits
    
    Parameters
    ----------
        lklc : lk.LightCurve() object
            must have time, flux, and cadenceno attributes
        transit_win_length: int
            reference size for transit mask, typically 3 x the maximum transit duration [number of cadences] 
        savgol_win_length: int
            size of Savitsky-Golay filter [number of cadences]
        transit_mask : array (bool)
            True for each cadence near transit
        combo_mask : array (bool)
            Same as transit mask, but wider around each transit to facilitate more careful detrending
        break_tolerance : int
            number of cadences considered a large gap in time
        polyorder : int
            polynomial order for Savitsky-Golay trend (default = 2)
        return_trend : bool
            True to return trend as well as flattened lightcurve
            
    Returns
    -------
        Flattened lightcurve (plus trend if return_trend==True)
    """
    # make lists to hold outputs
    flux_flat = []
    trend_flat = []    
    
    # identify gaps
    gap_locs = identify_gaps(lklc, transit_mask, break_tolerance)
    
    # break the data into contiguous segments and detrend
    for i, gloc in enumerate(gap_locs[:-1]):
        
        # these are the really short segments
        if gap_locs[i+1]-gap_locs[i] < transit_win_length:
            t = lklc.time[gap_locs[i]:gap_locs[i+1]]
            f = lklc.flux[gap_locs[i]:gap_locs[i+1]]
            m = transit_mask[gap_locs[i]:gap_locs[i+1]]
            
            try:
                pfit = np.polyfit(t[~m], f[~m], 1)
                simple_trend = np.polyval(pfit, t)
                
            except:
                try:
                    simple_trend = np.median(f[~m])
                except:
                    simple_trend = np.ones_like(f)
        
            flux_flat.append(f/simple_trend)
            trend_flat.append(simple_trend)
        
        
        # these are the segments with enough data to do real detrending
        else:
            # grab segments of time, flux, cadno, masks
            t = lklc.time[gap_locs[i]:gap_locs[i+1]]
            f = lklc.flux[gap_locs[i]:gap_locs[i+1]]
            c = lklc.cadenceno[gap_locs[i]:gap_locs[i+1]]

            m_transit = transit_mask[gap_locs[i]:gap_locs[i+1]]
            m_combo = combo_mask[gap_locs[i]:gap_locs[i+1]]
            
            
            # fill small gaps with white noise
            npts = c[-1]-c[0] + 1
            dt = np.min(t[1:]-t[:-1])
            
            t_interp = np.linspace(t.min(),t.max()+dt*3/2, npts)
            f_interp = np.ones_like(t_interp)
            c_interp = np.arange(c.min(), c.max()+1)

            data_exists = np.isin(c_interp, c)

            f_interp[data_exists] = f
            f_interp[~data_exists] = np.random.normal(loc=np.median(f), scale=np.std(f), \
                                                      size=np.sum(~data_exists))
            
            # apply Savitsky-Golay filter
            try:
                savgol_trend = sig.savgol_filter(f_interp, window_length=savgol_win_length, polyorder=polyorder)
                savgol_trend = savgol_trend[data_exists]
            except:
                try:
                    pfit = np.polyfit(t_interp[~m_transit], f_interp[~m_transit], polyorder)
                    savgol_trend = np.polyval(pfit, t_interp[data_exists])
                except:
                    savgol_trend = np.median(f_interp[data_exists])*np.ones(np.sum(data_exists))

            
            # replace points near transit (where S-G will give a bad detrending estimate)
            half_transit_win_length = int(np.floor(transit_win_length/2))
            half_savgol_win_length = int(np.floor(savgol_win_length/2))

            transit_trend = np.zeros_like(savgol_trend)
            bad = np.zeros_like(transit_trend, dtype='bool')

            for i in range(len(t)):
                if m_combo[i] == True:
                    
                    istart = int(np.max([0, i - half_savgol_win_length]))
                    iend   = int(np.min([len(t)+1, i + 1 + half_savgol_win_length]))

                    t_chunk = t[istart:iend]
                    f_chunk = f[istart:iend]
                    m_transit_chunk = m_transit[istart:iend]

                    if np.sum(~m_transit_chunk) > 1.2*half_savgol_win_length:
                        try:
                            pfit = np.polyfit(t_chunk[~m_transit_chunk], f_chunk[~m_transit_chunk], polyorder)
                            transit_trend[i] = np.polyval(pfit, t[i])
                        except:
                            bad[i] = True 

                    else:
                        bad[i] = True
                        
                        
            # put together componets for 1st estimate of full trend
            full_trend = np.copy(savgol_trend)
            full_trend[m_combo] = transit_trend[m_combo]
            
            
            # interpolate over poorly fit cadences or points in transit
            if np.sum(~bad)/len(bad) > 0.5:
                leftedge  = t < t[~bad].min()
                rightedge = t > t[~bad].max()
                edges = leftedge + rightedge

                _fxn = interp1d(t[~bad], full_trend[~bad])

                full_trend[~edges*bad] = _fxn(t[~edges*bad])
                full_trend[leftedge*bad] = full_trend[np.where(~bad)[0].min()]
                full_trend[rightedge*bad] = full_trend[np.where(~bad)[0].max()]


                # fix the edges of the segment
                final_trend = fix_edges(t, f, full_trend, m_transit, savgol_win_length)
         

                # save flattened flux and trend for output
                flux_flat.append(f/final_trend)
                trend_flat.append(final_trend)
            
            
            # if the various trending attempts failed, treat this segment as a very short segment
            else:
                m = m_transit
                
                try:
                    pfit = np.polyfit(t[~m], f[~m], 1)
                    simple_trend = np.polyval(pfit, t)

                except:
                    try:
                        simple_trend = np.median(f[~m])
                    except:
                        simple_trend = np.ones_like(f)

                flux_flat.append(f/simple_trend)
                trend_flat.append(simple_trend)


    
    
    # replace flux with flattened flux and return
    lklc.flux = np.hstack(flux_flat)
    
    if return_trend:
        return lklc, np.hstack(trend_flat)
    else:
        return lklc


    
def fix_edges(time, flux, trend, mask, savgol_win_length):
    """
    Docstring
    """    
    trend_fixed = np.copy(trend)
    
    # half window length (for convenience)
    half_savgol_win_length = int(np.floor(savgol_win_length/2))
    
    
    # avoid overfitting points near edges
    try:
        t_left = time[:savgol_win_length]
        f_left = flux[:savgol_win_length]
        m_left = mask[:savgol_win_length]

        trend_fixed[:half_savgol_win_length+1] = 0.5*(trend[half_savgol_win_length+1] \
                                                      + trend[:half_savgol_win_length+1])
    except:
        pass

    try:
        t_right = time[-(1+savgol_win_length):]
        f_right = flux[-(1+savgol_win_length):]
        m_right = mask[-(1+savgol_win_length):]

        trend_fixed[-(1+half_savgol_win_length):] = 0.5*(trend[-(1+half_savgol_win_length)] + \
                                                         trend[-(1+half_savgol_win_length):])
    except:
        pass


    # fix exponential ramp at beginning of each segment
    def _exp(theta, x):
        return theta[0] + theta[1]*np.exp(-x/np.abs(theta[2]))

    res_fxn = lambda theta, x, y: y - _exp(theta, x)

    # must have at least 3 "anchor points" to avoid extrapolation errors
    if np.sum(mask[:5]) < 3:

        t_left = time[:3*savgol_win_length]
        f_left = flux[:3*savgol_win_length]/trend_fixed[:3*savgol_win_length]
        m_left = mask[:3*savgol_win_length]

        bas = np.median(f_left[~m_left])
        amp = f_left[0] - f_left[-1]
        tau = t_left[np.min([half_savgol_win_length,len(t_left)-1])] - t_left[0]

        theta_in = np.array([bas, amp, tau])
        theta_out, success = op.leastsq(res_fxn, theta_in, \
                                        args=(t_left[~m_left]-t_left[0], f_left[~m_left]))

        exp_trend = _exp(theta_out, t_left-t_left[0])

        trend_fixed[:3*savgol_win_length] *= exp_trend

    return trend_fixed



def detrend_single_quarter(lklc, planets, transit_win_length_list, outlier_win_length, savgol_win_length, \
                            break_tolerance, polyorder=2, sigma_upper=5.0, sigma_lower=5.0, return_trend=False):
    """
    Docstring
    """
    # grab basic quantities
    dt = np.min(lklc.time[1:]-lklc.time[:-1])
    durs = np.zeros(len(planets), dtype='float64')
    
    for npl, p in enumerate(planets):
        durs[npl] = p.duration
    
    
    # do some simple cleanup
    lklc = remove_flagged_cadences(lklc)
    lklc = clip_outliers(lklc, kernel_size=outlier_win_length, sigma_upper=sigma_upper, sigma_lower=sigma_lower)
    
    
    # make masks
    transit_mask = np.zeros_like(lklc.time, dtype='bool')
    combo_mask   = np.zeros_like(transit_mask)

    for npl, p in enumerate(planets):
        transit_mask_size = transit_win_length_list[npl]/(p.duration/dt)/2
        combo_mask_size = transit_mask_size + savgol_win_length/(durs.max()/dt)/2
        
        transit_mask += make_transitmask(lklc.time, p.tts, p.duration, masksize=transit_mask_size)
        combo_mask += make_transitmask(lklc.time, p.tts, durs.max(), masksize=combo_mask_size)
    
    # flatten
    lklc = flatten(lklc, np.max(transit_win_length_list), savgol_win_length, transit_mask, combo_mask, break_tolerance)
    
    return lklc




class Planet:
    def __init__(self):
        self.dummy = None

    def calculate_stamp_chisq(self):
        '''
        Compare model_stamps, flux_stamps, and error_stamps to calcualte chisq for each transit
        '''
        mstamps = self.grab_stamps('model')
        fstamps = self.grab_stamps('flux')
        icov    = self.grab_icov()

        stamp_chisq = []
        j = 0
        for i, good in enumerate(self.quality):
            if good:
                y = mstamps[j]-fstamps[j]
                stamp_chisq.append(np.dot(y.T,np.dot(icov[j],y)))
                j += 1
            else:
                stamp_chisq.append(np.inf)

        self.stamp_chisq = np.array(stamp_chisq)

        return None


    def identify_good_transits(self, cover_fraction=0.7, chisq_sigma=5.0, verbose=True):
        '''
        Identify transits with sufficient coverage and non-outlier chisq

        cover_fraction: coverage threshold; eg. 0.7 will reject stamps with more than 70% of cadences missing (default=0.7)
        chisq_sigma: sigma threshold to reject stamps as poorly fit (default=5.0)
        verbose: boolean flag; 'True' to print results
        '''
        # determine locations of SC and LC data
        sc_loc = self.stamp_cadence == 'short'
        lc_loc = self.stamp_cadence == 'long'

        # flag stamps with sufficient coverage
        self.calculate_stamp_coverage()
        enough_pts = self.stamp_coverage > cover_fraction
        nonempty   = self.stamp_coverage > 0

        # count up points per stamp
        pts_per_stamp = []
        for t in self.time_stamps:
            pts_per_stamp.append(len(t))
        pts_per_stamp = np.array(pts_per_stamp)

        # flag stamps with unusually high chisq values (use pseudo-reduced-chisq)
        reject_chisq = np.zeros_like(self.tts, dtype='bool')

        if self.stamp_chisq is not None:
            X2u = self.stamp_chisq[~np.isinf(self.stamp_chisq)] / (pts_per_stamp[~np.isinf(self.stamp_chisq)])
            mad = astropy.stats.mad_std(X2u)
            med = np.median(X2u)
            reject_chisq[~np.isinf(self.stamp_chisq)] = np.abs(X2u-med)/mad > chisq_sigma
            reject_chisq[np.isinf(self.stamp_chisq)]  = True

        # print out results
        if verbose:
            print('%d out of %d transits rejected with high chisq' \
                  %(np.sum(reject_chisq[enough_pts]), np.sum(nonempty)))
            print('%d out of %d transits rejected with insufficient coverage' \
                  %(np.sum(~enough_pts[nonempty]), np.sum(nonempty)))

        # save the results
        self.quality = enough_pts * ~reject_chisq

        return None
    
    
    
    
def calculate_cdpp_snr():
    # Read in CDPP data
    cdpp = io.load_cdpp_data(PRIMARY_DIR + "Catalogs/keplerstellar_cdpp.csv")

    # do some data cleanup
    for k in cdpp.keys():
        cdpp[k] = np.asarray(cdpp[k])
        cdpp[k][cdpp[k] == ""] = "nan"


    for k in cdpp.keys():
        try:
            cdpp[k] = np.asarray(cdpp[k], dtype="float")
        except:
            pass

    cdpp["kepid"] = np.asarray(cdpp["kepid"], dtype="int")
    cdpp["nkoi"]  = np.asarray(cdpp["nkoi"], dtype="int")


    # pull cdpp durations and rms
    cdpp_dur, cdpp_rms = io.pull_cdpp_rms(cdpp, KIC)


    # interpolate CDPP and calculate single-transit SNR for each planet
    for npl, p in enumerate(planets):
        p.cdpp = interp1d(cdpp_dur, cdpp_rms, bounds_error=False, fill_value="extrapolate")(p.duration*24)
        p.snr  = p.depth*1e6/p.cdpp