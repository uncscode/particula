# For data processing of caps data

# flake8: noqa


# def caps_processing(
#         stream_size_distribution: Stream,
#         stream_caps: Stream,
#         truncation_bsca: bool = True,
#         truncation_interval_sec: int = 600,
#         truncation_interp: bool = True,
#         refractive_index: float = 1.45,
#         calibration_wet=1,
#         calibration_dry=1,
#         kappa_fixed: float = None,
# ):  # sourcery skip
#     """loader.
#     Function to process the CAPS data, and smps for kappa fitting, and then add
#     it to the datalake. Also applies truncation corrections to the CAPS data.

#     Args
#     ----------
#     datalake : object
#         DataLake object to add the processed data to.
#     truncation_bsca : bool, optional
#         Whether to calculate truncation corrections for the bsca data.
#         The default is True.
#     truncation_interval_sec : int, optional
#         The interval to calculate the truncation corrections over.
#         The default is 600. This can take around 10 sec per data point.
#     truncation_interp : bool, optional
#         Whether to interpolate the truncation corrections to the caps data.
#     refractive_index : float, optional
#         The refractive index of the aerosol. The default is 1.45.
#     calibration_wet : float, optional
#         The calibration factor for the wet data. The default is 1.
#     calibration_dry : float, optional
#         The calibration factor for the dry data. The default is 1.

#     Returns
#     -------
#     datalake : object
#         DataLake object with the processed data added.
#     """
#     # calc kappa and add to datalake
#     print('CAPS kappa_HGF fitting')

#     if kappa_fixed is None:
#         kappa_fit, _, _ = kappa_fitting_caps_data(
#             datalake=datalake,
#             truncation_bsca=False,
#             refractive_index=refractive_index
#         )
#     else:
#         kappa_len = len(
#             datalake.datastreams['CAPS_data'].return_time(
#                 datetime64=False))
#         kappa_fit = np.ones((kappa_len, 3)) * kappa_fixed

#     datalake.datastreams['CAPS_data'].add_processed_data(
#         data_new=kappa_fit.T,
#         time_new=datalake.datastreams['CAPS_data'].return_time(
#             datetime64=False),
#         header_new=['kappa_fit', 'kappa_fit_lower', 'kappa_fit_upper'],
#     )
#     orignal_average = datalake.datastreams['CAPS_data'].average_base_sec
#     orignal_average = datalake.datastreams['CAPS_data'].average_base_sec

#     # calc truncation corrections and add to datalake
#     print('CAPS truncation corrections')
#     if truncation_bsca:
#         datalake.reaverage_datastreams(
#             truncation_interval_sec,
#             stream_keys=['CAPS_data', 'smps_1D', 'smps_2D'],
#         )
#         # epoch_start=epoch_start,
#         # epoch_end=epoch_end

#         _, bsca_truncation_dry, bsca_truncation_wet = kappa_fitting_caps_data(
#             datalake=datalake,
#             truncation_bsca=truncation_bsca,
#             refractive_index=refractive_index
#         )

#         if truncation_interp:
#             interp_dry = interp1d(
#                 datalake.datastreams['CAPS_data'].return_time(
#                     datetime64=False),
#                 bsca_truncation_dry,
#                 kind='linear',
#                 fill_value='extrapolate'
#             )
#             interp_wet = interp1d(
#                 datalake.datastreams['CAPS_data'].return_time(
#                     datetime64=False),
#                 bsca_truncation_wet,
#                 kind='linear',
#                 fill_value='extrapolate'
#             )

#             time = datalake.datastreams['CAPS_data'].return_time(
#                 datetime64=False,
#                 raw=True
#             )
#             bsca_truncation_dry = interp_dry(time)
#             bsca_truncation_wet = interp_wet(time)
#         else:
#             time = datalake.datastreams['CAPS_data'].return_time(
#                 datetime64=False)

#         datalake.datastreams['CAPS_data'].add_processed_data(
#             data_new=bsca_truncation_dry.T,
#             time_new=time,
#             header_new=['bsca_truncation_dry'],
#         )
#         datalake.datastreams['CAPS_data'].add_processed_data(
#             data_new=bsca_truncation_wet.T,
#             time_new=time,
#             header_new=['bsca_truncation_wet'],
#         )
#     else:
#         bsca_truncation_wet = np.array([1])
#         bsca_truncation_dry = np.array([1])
#         time = datalake.datastreams['CAPS_data'].return_time(
#             datetime64=False,
#             raw=True
#         )

#     # index for b_sca wet and dry
#     index_dic = datalake.datastreams['CAPS_data'].return_header_dict()
#     index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

#     # check if raw in dict
#     if 'raw_b_sca_dry_CAPS_450nm[1/Mm]' in index_dic:
#         pass
#     else:
#         # save raw data
#         datalake.datastreams['CAPS_data'].add_processed_data(
#             data_new=datalake.datastreams['CAPS_data'].data_stream[
#                 index_dic['b_sca_wet_CAPS_450nm[1/Mm]'], :],
#             time_new=time,
#             header_new=['raw_b_sca_wet_CAPS_450nm[1/Mm]'],
#         )
#         datalake.datastreams['CAPS_data'].add_processed_data(
#             data_new=datalake.datastreams['CAPS_data'].data_stream[
#                 index_dic['b_sca_dry_CAPS_450nm[1/Mm]'], :],
#             time_new=time,
#             header_new=['raw_b_sca_dry_CAPS_450nm[1/Mm]'],
#         )
#         index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

#     datalake.datastreams['CAPS_data'].data_stream[
#         index_dic['b_sca_wet_CAPS_450nm[1/Mm]'], :] = \
#         datalake.datastreams['CAPS_data'].data_stream[
#             index_dic['raw_b_sca_wet_CAPS_450nm[1/Mm]'], :] \
#         * bsca_truncation_wet.T * calibration_wet

#     datalake.datastreams['CAPS_data'].data_stream[
#         index_dic['b_sca_dry_CAPS_450nm[1/Mm]'], :] = \
#         datalake.datastreams['CAPS_data'].data_stream[
#             index_dic['raw_b_sca_dry_CAPS_450nm[1/Mm]'], :] \
#         * bsca_truncation_dry.T * calibration_dry

#     datalake.datastreams['CAPS_data'].reaverage(
#         reaverage_base_sec=orignal_average
#     )  # updates the averages to the original value

#     return datalake


# def albedo_processing(
#     datalake,
#     keys: list = None
# ):
#     """
#     Calculates the albedo from the CAPS data and updates the datastream.

#     Args
#     ----------
#     datalake : object
#         DataLake object with the processed data added.

#     Returns
#     -------
#     datalake : object
#         DataLake object with the processed data added.
#     """

#     ssa_wet = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0] \
#         / datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0]
#     ssa_dry = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0] \
#         / datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0]
#     ssa_wet = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0] \
#         / datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0]
#     ssa_dry = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0] \
#         / datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0]

#     babs_wet = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0] \
#         - datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0]
#     babs_dry = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0] \
#         - datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0]
#     babs_wet = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0] \
#         - datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0]
#     babs_dry = datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0] \
#         - datalake.datastreams['CAPS_data'].return_data(
#         keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0]

#     time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)
#     time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)

#     datalake.datastreams['CAPS_data'].add_processed_data(
#         data_new=ssa_wet,
#         time_new=time,
#         header_new=['SSA_wet_CAPS_450nm[1/Mm]'],
#     )
#     datalake.datastreams['CAPS_data'].add_processed_data(
#         data_new=ssa_dry,
#         time_new=time,
#         header_new=['SSA_dry_CAPS_450nm[1/Mm]'],
#     )
#     datalake.datastreams['CAPS_data'].add_processed_data(
#         data_new=babs_wet,
#         time_new=time,
#         header_new=['b_abs_wet_CAPS_450nm[1/Mm]'],
#     )
#     datalake.datastreams['CAPS_data'].add_processed_data(
#         data_new=babs_dry,
#         time_new=time,
#         header_new=['b_abs_dry_CAPS_450nm[1/Mm]'],
#     )
#     return datalake


# def pass3_processing(
#         datalake: object,
#         babs_405_532_781=[1, 1, 1],
#         bsca_405_532_781=[1, 1, 1],
# ):
#     """
#     Processing PASS3 data applying the calibration factors
#     add the background correction

#     Args
#     ----------
#     datalake : object
#         DataLake object to add the processed data to.
#     babs_405_532_781 : list, optional
#         Calibration factors for the absorption channels. The default is [1,1,1]
#     bsca_405_532_781 : list, optional
#         Calibration factors for the scattering channels. The default is [1,1,1]

#     Returns
#     -------
#     datalake : object
#         DataLake object with the processed data added.
#     """
#     # index for b_sca wet and dry
#     index_dic = datalake.datastreams['pass3'].return_header_dict()
#     time = datalake.datastreams['pass3'].return_time(
#         datetime64=False,
#         raw=True
#     )
#     babs_list = ['b_abs405nm[1/Mm]', 'b_abs532nm[1/Mm]', 'b_abs781nm[1/Mm]']
#     bsca_list = ['b_sca405nm[1/Mm]', 'b_sca532nm[1/Mm]', 'b_sca781nm[1/Mm]']

#     if 'raw_b_abs405nm[1/Mm]' not in index_dic:
#         print('Copy raw babs Pass-3')
#         for babs in babs_list:
#             raw_name = f'raw_{babs}'

#             datalake.datastreams['pass3'].add_processed_data(
#                 data_new=datalake.datastreams['pass3'].data_stream[
#                     index_dic[babs], :],
#                 time_new=time,
#                 header_new=[raw_name],
#             )
#     if 'raw_b_sca405nm[1/Mm]' not in index_dic:
#         print('Copy raw bsca Pass-3')
#         for bsca in bsca_list:
#             raw_name = f'raw_{bsca}'

#             datalake.datastreams['pass3'].add_processed_data(
#                 data_new=datalake.datastreams['pass3'].data_stream[
#                     index_dic[bsca], :],
#                 time_new=time,
#                 header_new=[raw_name],
#             )

#     index_dic = datalake.datastreams['pass3'].return_header_dict()

#     # calibration loop babs.
#     print('Calibrated raw Pass-3')
#     for i, babs in enumerate(babs_list):
#         raw_name = f'raw_{babs}'
#         datalake.datastreams['pass3'].data_stream[index_dic[babs], :] = \
#             datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
#             * babs_405_532_781[i]
#     # calibration loop bsca
#     for i, bsca in enumerate(bsca_list):
#         raw_name = f'raw_{bsca}'
#         datalake.datastreams['pass3'].data_stream[index_dic[bsca], :] = \
#             datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
#             * bsca_405_532_781[i]

#     return datalake
