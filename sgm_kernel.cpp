#include "sgm_params.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/* Penalties */
const cost_t P1 = cost_t(10);
const cost_t P2 = cost_t(150);

static const cost_t INF_COST = cost_t(4095);

/* --------------------------------------------------------- */
/* Helper Function                                           */
/* --------------------------------------------------------- */
static constexpr int RIGHT_STRIPE_W = DISP + WIN - 1;

static inline void update_line_buffers(
		xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
		xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
		int c,
		pix_t pL,
		pix_t pR)
{
#pragma HLS INLINE
    bufL.shift_up(c);
    bufL.insert_bottom(pL, c);

    bufR.shift_up(c);
    bufR.insert_bottom(pR, c);
}

static void update_sliding_windows(
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
		int c,
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		int& right_wr)
{
#pragma HLS INLINE

	ShiftLeftWin:
	    for (int wy = 0; wy < WIN; ++wy)
	    {
		#pragma HLS UNROLL
	        for (int wx = 0; wx < WIN - 1; ++wx)
	        {
		#pragma HLS UNROLL
	            leftWin[wy][wx] = leftWin[wy][wx + 1];
	        }
	    }

	InsertLeftCol:
		for (int wy = 0; wy < WIN; ++wy)
		{
	    #pragma HLS UNROLL
			leftWin[wy][WIN - 1] = bufL.getval(wy, c);
	    }

		right_wr++;
		if(right_wr == RIGHT_STRIPE_W)
			right_wr = 0;

	ShiftRightStripe:
		for (int wy = 0; wy < WIN; ++wy)
		{
		#pragma HLS UNROLL
			rightStripe[wy][right_wr] = bufR.getval(wy, c);
		}

}

static void compute_sad_cost_vector(
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		int right_wr,
	    cost_t curCost[DISP])
{
#pragma HLS INLINE off

	SAD_Disparity:
	for (int d = 0; d < DISP; ++d)
	{
	#pragma HLS PIPELINE II=1
		cost_t sum = 0;

    SAD_WinY:
        for (int wy = 0; wy < WIN; ++wy)
        {
		#pragma HLS UNROLL

        SAD_WinX:
            for (int wx = 0; wx < WIN; ++wx)
            {
			#pragma HLS UNROLL
            	int logicalIndex = RIGHT_STRIPE_W - WIN - d + wx;

            	int physIndex = right_wr + 1 + logicalIndex;
            	if (physIndex >= RIGHT_STRIPE_W)
            			physIndex -= RIGHT_STRIPE_W;

                pix_t lpx = leftWin[wy][wx];
                pix_t rpx = rightStripe[wy][physIndex];

                sum += absdiff(lpx, rpx);
            }
        }
        curCost[d] = sum;
	}
}

static disp_t aggregate_paths_and_select(
    const cost_t curCost[DISP],
    const cost_t prevCostL[DISP],
    const cost_t prevCostT_col[DISP],
    cost_t minPrevLR,
    cost_t minPrevTB,
    cost_t aggLR_arr[DISP],
    cost_t aggTB_arr[DISP],
    cost_t aggCost[DISP],
	cost_t& newMinLR,
	cost_t& newMinTB)
{
#pragma HLS INLINE off

    cost_t bestCost = INF_COST;
    disp_t bestDisp = 0;

    cost_t runMinLR = INF_COST;
    cost_t runMinTB = INF_COST;

AggregationLoop:
    for (int d = 0; d < DISP; d++)
    {
	#pragma HLS PIPELINE II = 1
        cost_t p0_LR = prevCostL[d];
        cost_t p1_LR = (d > 0) ? sat12(prevCostL[d - 1] + P1) : INF_COST;
        cost_t p2_LR = (d < DISP - 1) ? sat12(prevCostL[d + 1] + P1) : INF_COST;
        cost_t p3_LR = sat12(minPrevLR + P2);

        cost_t minLR = p0_LR;
        if (p1_LR < minLR) minLR = p1_LR;
        if (p2_LR < minLR) minLR = p2_LR;
        if (p3_LR < minLR) minLR = p3_LR;

        cost_t aggLR = sat12(curCost[d] + minLR - minPrevLR);
        aggLR_arr[d] = aggLR;

        cost_t p0_TB = prevCostT_col[d];
        cost_t p1_TB = (d > 0) ? sat12(prevCostT_col[d - 1] + P1) : INF_COST;
        cost_t p2_TB = (d < DISP - 1) ? sat12(prevCostT_col[d + 1] + P1) : INF_COST;
        cost_t p3_TB = sat12(minPrevTB + P2);

        cost_t minTB = p0_TB;
        if (p1_TB < minTB) minTB = p1_TB;
        if (p2_TB < minTB) minTB = p2_TB;
        if (p3_TB < minTB) minTB = p3_TB;

        cost_t aggTB = sat12(curCost[d] + minTB - minPrevTB);
        aggTB_arr[d] = aggTB;

        if(aggLR < runMinLR) runMinLR = aggLR;
        if(aggTB < runMinTB) runMinTB = aggTB;

        cost_t sum2 = sat12(aggLR + aggTB);
        aggCost[d] = sum2;

        if (sum2 < bestCost)
        {
            bestCost = sum2;
            bestDisp = disp_t(d);
        }
    }
    newMinLR = runMinLR;
    newMinTB = runMinTB;

    return bestDisp;
}

static void commit_prev_costs(
    cost_t prevCostL[DISP],
    cost_t prevCostT_col[DISP],
    const cost_t aggLR_arr[DISP],
    const cost_t aggTB_arr[DISP])
{
#pragma HLS INLINE off

CopyPrevLR:
    for (int d = 0; d < DISP; ++d)
    {
	#pragma HLS UNROLL
        prevCostL[d]    = aggLR_arr[d];
        prevCostT_col[d] = aggTB_arr[d];
    }
}
struct CostPacket
{
	bool valid;
	cost_t curCost[DISP];
};

static CostPacket col_frontend(
	    hls::stream<pix_t>& left,
	    hls::stream<pix_t>& right,
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
	    int r,
	    int c,
	    int cx,
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		int& right_wr)
{
#pragma HLS INLINE off
	CostPacket pkt;
	pkt.valid = false;

    pix_t pL = left.read();
    pix_t pR = right.read();

	update_line_buffers(bufL, bufR, c, pL, pR);
	update_sliding_windows(bufL, bufR, c, leftWin, rightStripe, right_wr);

	const bool interior =
	    (r >= WIN - 1) &&
	    (c >= (DISP - 1) + 2* cx) &&
	    (c < IMG_W);

    if (interior)
    {
    	compute_sad_cost_vector(leftWin, rightStripe, right_wr, pkt.curCost);
    	pkt.valid = true;
    }
    else
    {
    	pkt.valid = false;
    }
    return pkt;
}

static pix_t col_backend(
		const CostPacket& pkt,
		cost_t prevCostL[DISP],
		cost_t prevCostT_col[DISP],
		cost_t aggLR_arr[DISP],
		cost_t aggTB_arr[DISP],
		cost_t aggCost[DISP],
		cost_t& minPrevLR,
		cost_t& minPrevTB)
{
#pragma HLS INLINE off
	pix_t outDisp = 0;

	if (pkt.valid)
	{
        cost_t newMinLR = INF_COST;
        cost_t newMinTB = INF_COST;

        disp_t bestDisp = aggregate_paths_and_select(
            pkt.curCost,
            prevCostL,
			prevCostT_col,
            minPrevLR,
            minPrevTB,
            aggLR_arr,
            aggTB_arr,
            aggCost,
			newMinLR,
			newMinTB);

        commit_prev_costs(
            prevCostL,
			prevCostT_col,
            aggLR_arr,
            aggTB_arr);

        minPrevLR = newMinLR;
        minPrevTB = newMinTB;
        outDisp = bestDisp;
	}
    return outDisp;
}

/* --------------------------------------------------------- */
/* Top kernel                                                */
/* --------------------------------------------------------- */

void sgm_kernel(hls::stream<pix_t>& left,
                hls::stream<pix_t>& right,
                hls::stream<pix_t>& disp)
{
#pragma HLS INTERFACE axis         port=left   register
#pragma HLS INTERFACE axis         port=right  register
#pragma HLS INTERFACE axis         port=disp   register
#pragma HLS INTERFACE ap_ctrl_none port=return
//#pragma HLS DATAFLOW

    /* Line buffers for the left & right images */
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufL;
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufR;

    InitBuf:
    for (int wy = 0; wy < WIN; ++wy)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            bufL.val[wy][c] = 0;
            bufR.val[wy][c] = 0;
        }
    }

    /* Cost arrays */
    static cost_t prevCostL[DISP];
#pragma HLS ARRAY_PARTITION variable=prevCostL complete dim=1

    static cost_t prevCostT[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostT type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostT complete dim=2

    static cost_t aggCost[DISP];
#pragma HLS ARRAY_PARTITION variable=aggCost complete dim=1

    static cost_t aggLR_arr[DISP];
    static cost_t aggTB_arr[DISP];
#pragma HLS ARRAY_PARTITION variable=aggLR_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=aggTB_arr complete dim=1

    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    static cost_t minPrevT[IMG_W];

    /* center offset */
    const int cx = WIN >> 1;

Row:
    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;

    	cost_t minPrevLR = 0;

        /* Reset aggregation for new row */
    ResetCosts:
        for (int d = 0; d < DISP; d++)
        {
		#pragma HLS UNROLL factor=2
            prevCostL[d] = cost_t(0);
        }

        if(r == 0)
        {
        InitTBRow:
			for (int c = 0; c < IMG_W; ++c)
			{
			#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W
				InitTBRowD:
				for (int d = 0; d < DISP; ++d)
				{
				#pragma HLS UNROLL factor=2
					prevCostT[c][d] = cost_t(0);
				}
			}
        }

    	InitLeftWin:
    	for (int wy = 0; wy < WIN; ++wy)
    	{
    	    for (int wx = 0; wx < WIN; ++wx)
    	    {
    	        leftWin[wy][wx] = 0;
    	    }
    	}

    	InitRightStripe:
    	for (int wy = 0; wy < WIN; ++wy)
    	{
    	    for (int k = 0; k < RIGHT_STRIPE_W; ++k)
    	    {
    	        rightStripe[wy][k] = 0;
    	    }
    	}

    	for (int c = 0; c < IMG_W; ++c)
    	{
    	//#pragma HLS PIPELINE II=16
    	#pragma HLS DEPENDENCE variable=bufL inter false
    	#pragma HLS DEPENDENCE variable=bufR inter false

    		CostPacket pkt = col_frontend(
    				left,
					right,
					bufL,
					bufR,
					r,
					c,
					cx,
    				leftWin,
					rightStripe,
					right_wr);

    		int out_c = c - cx;
    		if(out_c >= 0)
    		{
    			pix_t outDisp = col_backend(
    					pkt,
    					prevCostL,
						prevCostT[out_c],
						aggLR_arr,
						aggTB_arr,
						aggCost,
						minPrevLR,
						minPrevT[out_c]);

    				disp.write(outDisp);
    		}
    	}
        for (int t = 0; t < cx; ++t)
        {
            disp.write(0);
        }
    }
}
