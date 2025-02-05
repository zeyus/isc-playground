import streamlit as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from timeit import default_timer

st.title('ISC Playground')
st.subheader('Updated 2025-02-05 10:37')
st.link_button('Source code', 'https://github.com/zeyus/isc-playground')


if 'conditions' not in st.session_state:
    st.session_state.conditions = ['all']


with st.sidebar:
    with st.expander('💻 Simulation parameters'):
        n_subj  = st.select_slider('Number of subjects', options=range(1, 11), value=5)
        n_chan  = st.select_slider('Number of channels', options=range(1, 65), value=2)
        sample_rate = st.select_slider('Sample rate', options=[5, 10, 50, 100, 250, 500, 1000], value=5)
        duration = st.text_input('Duration (seconds)', '10')

        # and of the signal, how many are correlated?
        corr_disabled=False
        if n_subj < 2:
            corr_disabled = True
        n_correlated = st.select_slider('Number of correlated signal subjects', options=range(0, max(3, n_subj+1)), value=min(2,n_subj), disabled=corr_disabled)
    with st.expander('📡 Signal parameters'):
        # signal parameters
        signal_type = st.selectbox('Signal type', ['sine', 'square', 'sawtooth', 'triangle', 'random'])
        signal_freq = st.slider('Signal frequency', min_value=1, max_value=10, value=1)
        signal_amp = st.slider('Signal amplitude', min_value=0.1, max_value=1.0, value=0.5)
        signal_phase = st.slider('Signal phase', min_value=0., max_value=2*np.pi, value=0.)
        signal_noise = st.slider('Signal noise', min_value=0., max_value=0.1, value=0.01)
    with st.expander('🔮 Random parameters'):
        # random parameters
        random_noise = st.slider('Random noise', min_value=0., max_value=0.1, value=0.01)
    with st.expander('🌀 Correlation parameters'):
        # correlation parameters
        correlation = st.slider('Correlation', min_value=0., max_value=1., value=0.5)
    with st.expander('📊 Data Parameters'):
        st.subheader('Conditions')
        
        def add_cond(cond:str):
            st.session_state.conditions.append(cond)
        
        def remove_cond(cond:str):
            st.session_state.conditions.remove(cond)
        
        for cond in st.session_state.conditions:
            col1, col2 = st.columns([4,1], vertical_alignment="center")
            with col1:
                st.write(f'{cond}')
            with col2:
                st.button(
                    '',
                    key=f"del-{cond}",
                    icon=":material/delete:",
                    type="primary",
                    disabled=cond=='all',
                    on_click=remove_cond,
                    args=(cond,)
                )
            st.divider()
        col1, col2 = st.columns([4,1], vertical_alignment="bottom")
        new_condition = ''
        with col1:
            new_condition = st.text_input('New condition name', key='new_condition')
        with col2:
            st.button('', key="add-cond", icon=":material/add:", type="primary", on_click=add_cond, args=(new_condition,))
        
        # list subjects and add / remove them from conditions via checkboxes
        # cols = st.columns(len(st.session_state.conditions))
        
        # add headers
        # with cols[0]:
        #     st.write('Subj.')
        # for j, cond in enumerate(st.session_state.conditions):
        #     if cond == 'all':
        #         continue
        #     with cols[j]:
        #         st.write(cond)

        # cols = st.columns(len(st.session_state.conditions), vertical_alignment="top", border=False, gap="small")
        # for i in range(n_subj):
        #     with cols[0]:
        #         st.write(f'S{i+1}')
        #     for j, cond in enumerate(st.session_state.conditions):
        #         if cond == 'all':
        #             continue
        #         with cols[j]:
        #             st.checkbox(f'subject {i} in {cond}', key=f'subj-{i}-{cond}', label_visibility="hidden")
        tabs = st.tabs(st.session_state.conditions)
        for t,cond in enumerate(st.session_state.conditions):
            with tabs[t]:
                for i in range(n_subj):
                    if cond == 'all':
                        st.checkbox(f'subject {i}', key=f'subj-{i}-{cond}', value=True, disabled=True)
                    else:    
                        st.checkbox(f'subject {i}', key=f'subj-{i}-{cond}')



# generate data
@st.cache_data
def generate_data():
    time = np.arange(0, int(duration)*sample_rate, 1)
    data = np.zeros((n_subj, n_chan, len(time)))

    n_random = n_subj - n_correlated
    # generate random data
    for i in range(n_random):
        data[i] = np.random.rand(n_chan, len(time)) * random_noise

    # next generated the correlated subject data
    # starting with a base signal using the desired signal type
    base_signal = np.zeros((n_chan, len(time)))
    if signal_type == 'sine':
        base_signal += signal_amp * np.sin(2*np.pi*signal_freq*time/sample_rate + signal_phase)
    elif signal_type == 'square':
        base_signal += signal_amp * sp.signal.square(2*np.pi*signal_freq*time/sample_rate + signal_phase)
    elif signal_type == 'sawtooth':
        base_signal += signal_amp * sp.signal.sawtooth(2*np.pi*signal_freq*time/sample_rate + signal_phase)
    elif signal_type == 'triangle':
        base_signal += signal_amp * sp.signal.sawtooth(2*np.pi*signal_freq*time/sample_rate + signal_phase, width=0.5)
    elif signal_type == 'random':
        base_signal += np.random.rand(n_chan, len(time)) * signal_amp

    # add noise to the base signal
    base_signal += np.random.rand(n_chan, len(time)) * signal_noise

    if n_correlated > 1:
        # generate correlated data
        data[n_random] = base_signal
        for i in range(n_random+1, n_subj):
            # now create remaining correlated data
            data[i] = base_signal * correlation + np.random.rand(n_chan, len(time)) * (1-correlation)

    return data, time

# Allow user to assign subjects to conditions




# from: https://github.com/ML-D00M/ISC-Inter-Subject-Correlations/blob/main/Python/ISC.py
@st.cache_data
def train_cca(data):
    """Run Correlated Component Analysis on your training data.

        Parameters:
        ----------
        data : dict
            Dictionary with keys are names of conditions and values are numpy
            arrays structured like (subjects, channels, samples).
            The number of channels must be the same between all conditions!

        Returns:
        -------
        W : np.array
            Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
            correlation the most.
        ISC : np.array
            Inter-subject correlation sorted in descending order

    """

    start = default_timer()

    C = len(data.keys())
    st.write(f'train_cca - calculations started. There are {C} conditions')

    gamma = 0.1
    Rw, Rb = 0, 0
    for cond in data.values():
        N, D, T, = cond.shape
        st.write(f'Condition has {N} subjects, {D} sensors and {T} samples')
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :]
                           for i in range(0, N)], axis=0)

        # Rb
        Rb = Rb + np.mean([Rij[i, j, :, :]
                           for i in range(0, N)
                           for j in range(0, N) if i != j], axis=0)

    # Divide by number of condition
    Rw, Rb = Rw/C, Rb/C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]

    stop = default_timer()

    st.write(f'Elapsed time: {round(stop - start)} seconds.')
    return W, ISC

@st.cache_data
def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.

        Parameters:
        ----------
        X : ndarray
            3-D numpy array structured like (subject, channel, sample)
        W : ndarray
            Spatial filters.
        fs : int
            Frequency sampling.
        Returns:
        -------
        ISC : ndarray
            Inter-subject correlations values are sorted in descending order.
        ISC_persecond : ndarray
            Inter-subject correlations values per second where first row is the most correlated.
        ISC_bysubject : ndarray
            Description goes here.
        A : ndarray
            Scalp projections of ISC.
    """

    start = default_timer()
    st.write('apply_cca - calculations started')

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 5
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    Rw = np.mean([Rij[i, i, :, :]
                  for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean([Rij[i, j, :, :]
                  for i in range(0, N)
                  for j in range(0, N) if i != j], axis=0)

    # ISCs
    ISC = np.sort(np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W))[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)

    # ISC by subject
    st.write('by subject is calculating')
    ISC_bysubject = np.empty((D, N))

    for subj_k in range(0, N):
        Rw, Rb = 0, 0
        Rw = np.mean([Rw + 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)
        Rb = np.mean([Rb + 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)

    # ISC per second
    st.write('by persecond is calculating')
    ISC_persecond = np.empty((D, int(T / fs) + 1))
    window_i = 0

    for t in range(0, T, fs):

        Xt = X[:, t:t+window_sec*fs]
        Rij = np.cov(Xt)
        Rw = np.mean([Rij[i:i + D, i:i + D]
                      for i in range(0, D * N, D)], axis=0)
        Rb = np.mean([Rij[i:i + D, j:j + D]
                      for i in range(0, D * N, D)
                      for j in range(0, D * N, D) if i != j], axis=0)

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
        window_i += 1

    stop = default_timer()
    st.write(f'Elapsed time: {round(stop - start)} seconds.')

    return ISC, ISC_persecond, ISC_bysubject, A

@st.cache_data
def plot_isc(isc_all):
    # plot ISC as a bar chart
    plot1 = plt.figure()
    # get number of components
    n_comp = len(isc_all[list(isc_all.keys())[0]]['ISC'])
    comp1 = [cond['ISC'][0] for cond in isc_all.values()]
    if n_comp > 1:
        comp2 = [cond['ISC'][1] for cond in isc_all.values()]
    if n_comp > 2:
        comp3 = [cond['ISC'][2] for cond in isc_all.values()]
    barWidth = 0.2
    r1 = np.arange(len(comp1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, comp1, color='gray', width=barWidth, edgecolor='white', label='Comp1')
    if n_comp > 1:
        plt.bar(r2, comp2, color='green', width=barWidth, edgecolor='white', label='Comp2')
    if n_comp > 2:
        plt.bar(r3, comp3, color='green', width=barWidth, edgecolor='white', label='Comp3')
    plt.xticks([r + barWidth for r in range(len(comp1))], isc_all.keys())
    plt.ylabel('ISC', fontweight='bold')
    plt.title('ISC for each condition')
    plt.legend()


    plot2 = plt.figure()
    # plot ISC_persecond
    for cond in isc_all.values():
        for comp_i in range(0, min(n_comp, 3)):
            plt.subplot(3, 1, comp_i+1)
            plt.title(f'Component {comp_i+1}', loc='right')
            plt.plot(cond['ISC_persecond'][comp_i])
            # plt.legend(isc_all.keys())
            plt.xlabel('Time (s)')
            plt.ylabel('ISC')
    

    return plot1, plot2



def run_cca(data_dict):
    [W, ISC_overall] = train_cca(data_dict)

    # plot spatial filter weights
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(W, ax=ax)
    ax.set_title('Spatial filter weights')
    st.pyplot(fig)

    # apply CCA
    isc_results=dict()
    for cond_key, cond_values in data_dict.items():
        isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(cond_values, W, sample_rate)))

    plot1, plot2 = plot_isc(isc_results)

    st.pyplot(plot1)
    st.pyplot(plot2)



# plot data
data, time = generate_data()
fig, ax = plt.subplots(n_subj, figsize=(10, 15))
for i in range(n_subj):
    # plot all channels for each subject
    for j in range(n_chan):
        ax[i].plot(time, data[i, j, :])
    ax[i].set_ylabel(f'S{i+1}')
fig.supxlabel('Time (ms)')
fig.supylabel('Amplitude')

plt.tight_layout()
st.pyplot(fig)

# plot correlation matrix
corr = np.zeros((n_subj, n_subj))
for i in range(n_subj):
    for j in range(n_subj):
        corr[i, j] = np.corrcoef(data[i].flatten(), data[j].flatten())[0, 1]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.heatmap(corr, ax=ax)
ax.set_title('Correlation matrix')
st.pyplot(fig)


# plot power spectrum
fig, ax = plt.subplots(n_subj, figsize=(10, 5))
for i in range(n_subj):
    # plot all channels for each subject
    for j in range(n_chan):
        f, Pxx = sp.signal.welch(data[i, j, :], fs=sample_rate, nperseg=256)
        ax[i].plot(f, Pxx)
    ax[i].set_ylabel(f'S{i+1}')
fig.supxlabel('Frequency (Hz)')
fig.supylabel('Power')


st.pyplot(fig)

st.write(data.shape)
    
data_dict = dict()
for cond in st.session_state.conditions:
    if cond == 'all':
        data_dict[cond] = data
        continue

    included_subj = [i for i in range(n_subj) if st.session_state[f'subj-{i}-{cond}']]
    if len(included_subj) > 2:
        data_dict[cond] = data[included_subj]

run_cca(data_dict)
