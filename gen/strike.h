#pragma once
#include <stdint.h>
typedef struct gs_stack gs_stack;
#pragma pack(push, 1)
typedef struct { uint8_t *base; int64_t len; } gs_rhdr;
typedef struct { gs_rhdr *hdr; gs_stack *stk; } gs_rref;
typedef struct { gs_rhdr *hdr; gs_stack *stk; gs_rhdr *fl; gs_stack *flstk; } gs_pref;

typedef struct { double e[23]; } a23_f64;

typedef struct { double e[2]; } a2_f64;

typedef struct { double e[5]; } a5_f64;

typedef struct { double e[180000]; } a180000_f64;

typedef struct { double e[60000]; } a60000_f64;

typedef struct { int64_t e[180000]; } a180000_i64;

typedef struct { int32_t e[60000]; } a60000_i32;

typedef struct { int32_t e[131072]; } a131072_i32;

typedef struct { double e[72000]; } a72000_f64;

typedef struct { double e[24000]; } a24000_f64;

typedef struct { int32_t e[24000]; } a24000_i32;

typedef struct { int32_t e[96000]; } a96000_i32;

typedef struct { double e[360000]; } a360000_f64;

typedef struct { double e[120000]; } a120000_f64;

typedef struct { double e[48000]; } a48000_f64;

typedef struct { int64_t e[72000]; } a72000_i64;

typedef struct { float e[72000]; } a72000_f32;

typedef struct { float e[180000]; } a180000_f32;

typedef struct { float e[60000]; } a60000_f32;

typedef struct { float e[24000]; } a24000_f32;

typedef struct { double e[3]; } a3_f64;

typedef struct { uint8_t *data; int64_t len; } sl_u8;

typedef struct { int64_t e[4]; } a4_i64;

typedef struct { double e[4]; } a4_f64;

typedef struct ByteSet_g ByteSet_g;
typedef struct { uint8_t e[256]; } a256_b;

typedef struct { uint8_t e[3]; } a3_u8;

typedef struct { uint8_t e[16]; } a16_u8;

struct ByteSet_g {
    a256_b members_g;
    uint8_t kind_g;
    uint8_t invert_g;
    a3_u8 bytes_g;
    a16_u8 lo_g;
    a16_u8 hi_g;
};

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif
void goose_init(int argc, char **argv);
int32_t setup(int32_t gs_export_arg0, int32_t gs_export_arg1, double gs_export_arg2, double gs_export_arg3);
int32_t advance(int32_t gs_export_arg0_2, double gs_export_arg1_2);
int32_t step_phase(double gs_export_arg0_3);
double get_time(void);
double get_dt(void);
int32_t get_steps(void);
double get_t_contact(void);
int32_t get_np(void);
int32_t get_nn(void);
int32_t get_ne(void);
double get_particle_mass(void);
double get_spacing(void);
double get_smoothing(void);
double get_bird_d(void);
double get_bird_l(void);
double get_bird_v(void);
double get_contact_force(void);
double get_center_force(void);
double get_sensor_r(void);
double get_impulse(void);
double get_panel_work(void);
double get_hg_work(void);
double get_bird_work(void);
int32_t get_failed(void);
int32_t get_failed_skin(void);
double get_max_ep(void);
double get_max_defl(void);
double get_eps_fail(int32_t gs_export_arg0_4);
double get_bird_ke(void);
double get_panel_ke(void);
double get_panel_mass(void);
int32_t fill_render(void);
const uint8_t * buf_nodes(int64_t *gs_export_len_33);
const uint8_t * buf_particles(int64_t *gs_export_len_34);
const uint8_t * buf_pressure(int64_t *gs_export_len_35);
const uint8_t * buf_strain(int64_t *gs_export_len_36);
const uint8_t * buf_elements(int64_t *gs_export_len_37);
#ifdef __cplusplus
}
#endif
