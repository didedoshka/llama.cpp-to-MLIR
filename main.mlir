module {
	func.func @compute_graph_forward() {
		%inp_embd = "GET_ROWS"(%token_embd.weight, %leaf_2) : (tensor<288x32000x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-0 = "RMS_NORM"(%inp_embd) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-0 = "MUL"(%norm-0, %blk.0.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-0 = linalg.matmul ins(%blk.0.attn_q.weight, %attn_norm-0 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-0 = "MUL_MAT"(%blk.0.attn_q.weight, %attn_norm-0) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-0__reshaped_ = "RESHAPE"(%Qcur-0) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-0 = "ROPE"(%Qcur-0__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-0 = linalg.matmul ins(%blk.0.attn_v.weight, %attn_norm-0 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-0 = "MUL_MAT"(%blk.0.attn_v.weight, %attn_norm-0) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-0 = "RESHAPE"(%Vcur-0) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-0 = linalg.matmul ins(%blk.0.attn_k.weight, %attn_norm-0 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-0 = "MUL_MAT"(%blk.0.attn_k.weight, %attn_norm-0) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-0__reshaped_ = "RESHAPE"(%Kcur-0) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-0 = "ROPE"(%Kcur-0__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-0__view_ = "VIEW"(%Kcur-0) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l0__view_ = "SET_ROWS"(%Kcur-0__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-0__view_ = "VIEW"(%Vcur-0) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l0__view_ = "SET_ROWS"(%Vcur-0__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-0__view_ = "VIEW"(%Qcur-0) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-0__view___permuted_ = "PERMUTE"(%Qcur-0__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l0__view_ = "VIEW"(%cache_k_l0) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l0__view___permuted_ = "PERMUTE"(%cache_k_l0__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l0__view_ = "VIEW"(%cache_v_l0) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l0__view___permuted_ = "PERMUTE"(%cache_v_l0__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__copy_ = "CPY"(%leaf_11, %__copy_) : (tensor<256x3x1x1xf32>, tensor<256x3x1x1xf32>) -> tensor<256x3x1x1xf32>

		%__fattn__-0 = "FLASH_ATTN_EXT"(%Qcur-0__view___permuted_, %cache_k_l0__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-0 = "RESHAPE"(%__fattn__-0) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-0 = linalg.matmul ins(%blk.0.attn_output.weight, %kqv_out-0 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-0 = "MUL_MAT"(%blk.0.attn_output.weight, %kqv_out-0) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_inp-0 = linalg.add ins(%attn_out-0, %inp_embd : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_inp-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-0 = "ADD"(%attn_out-0, %inp_embd) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-0 = "RMS_NORM"(%ffn_inp-0) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_norm-0 = "MUL"(%norm-0, %blk.0.ffn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_gate-0_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_gate-0 = linalg.matmul ins(%blk.0.ffn_gate.weight, %ffn_norm-0 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_gate-0_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_gate-0 = "MUL_MAT"(%blk.0.ffn_gate.weight, %ffn_norm-0) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-0_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_up-0 = linalg.matmul ins(%blk.0.ffn_up.weight, %ffn_norm-0 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_up-0_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-0 = "MUL_MAT"(%blk.0.ffn_up.weight, %ffn_norm-0) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_swiglu-0 = "GLU"(%ffn_gate-0, %ffn_up-0) : (tensor<768x3x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_out-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_out-0 = linalg.matmul ins(%blk.0.ffn_down.weight, %ffn_swiglu-0 : tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>)
			outs(%ffn_out-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_out-0 = "MUL_MAT"(%blk.0.ffn_down.weight, %ffn_swiglu-0) : (tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-0_init = tensor.empty() : tensor<288x3x1x1xf32>
		%l_out-0 = linalg.add ins(%ffn_out-0, %ffn_inp-0 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%l_out-0_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-0 = "ADD"(%ffn_out-0, %ffn_inp-0) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-1 = "RMS_NORM"(%l_out-0) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-1 = "MUL"(%norm-1, %blk.1.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-1 = linalg.matmul ins(%blk.1.attn_q.weight, %attn_norm-1 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-1 = "MUL_MAT"(%blk.1.attn_q.weight, %attn_norm-1) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-1__reshaped_ = "RESHAPE"(%Qcur-1) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-1 = "ROPE"(%Qcur-1__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-1 = linalg.matmul ins(%blk.1.attn_v.weight, %attn_norm-1 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-1 = "MUL_MAT"(%blk.1.attn_v.weight, %attn_norm-1) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-1 = "RESHAPE"(%Vcur-1) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-1 = linalg.matmul ins(%blk.1.attn_k.weight, %attn_norm-1 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-1 = "MUL_MAT"(%blk.1.attn_k.weight, %attn_norm-1) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-1__reshaped_ = "RESHAPE"(%Kcur-1) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-1 = "ROPE"(%Kcur-1__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-1__view_ = "VIEW"(%Kcur-1) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l1__view_ = "SET_ROWS"(%Kcur-1__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-1__view_ = "VIEW"(%Vcur-1) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l1__view_ = "SET_ROWS"(%Vcur-1__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-1__view_ = "VIEW"(%Qcur-1) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-1__view___permuted_ = "PERMUTE"(%Qcur-1__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l1__view_ = "VIEW"(%cache_k_l1) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l1__view___permuted_ = "PERMUTE"(%cache_k_l1__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l1__view_ = "VIEW"(%cache_v_l1) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l1__view___permuted_ = "PERMUTE"(%cache_v_l1__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__fattn__-1 = "FLASH_ATTN_EXT"(%Qcur-1__view___permuted_, %cache_k_l1__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-1 = "RESHAPE"(%__fattn__-1) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-1 = linalg.matmul ins(%blk.1.attn_output.weight, %kqv_out-1 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-1 = "MUL_MAT"(%blk.1.attn_output.weight, %kqv_out-1) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_inp-1 = linalg.add ins(%attn_out-1, %l_out-0 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_inp-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-1 = "ADD"(%attn_out-1, %l_out-0) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-1 = "RMS_NORM"(%ffn_inp-1) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_norm-1 = "MUL"(%norm-1, %blk.1.ffn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_gate-1_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_gate-1 = linalg.matmul ins(%blk.1.ffn_gate.weight, %ffn_norm-1 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_gate-1_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_gate-1 = "MUL_MAT"(%blk.1.ffn_gate.weight, %ffn_norm-1) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-1_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_up-1 = linalg.matmul ins(%blk.1.ffn_up.weight, %ffn_norm-1 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_up-1_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-1 = "MUL_MAT"(%blk.1.ffn_up.weight, %ffn_norm-1) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_swiglu-1 = "GLU"(%ffn_gate-1, %ffn_up-1) : (tensor<768x3x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_out-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_out-1 = linalg.matmul ins(%blk.1.ffn_down.weight, %ffn_swiglu-1 : tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>)
			outs(%ffn_out-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_out-1 = "MUL_MAT"(%blk.1.ffn_down.weight, %ffn_swiglu-1) : (tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-1_init = tensor.empty() : tensor<288x3x1x1xf32>
		%l_out-1 = linalg.add ins(%ffn_out-1, %ffn_inp-1 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%l_out-1_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-1 = "ADD"(%ffn_out-1, %ffn_inp-1) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-2 = "RMS_NORM"(%l_out-1) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-2 = "MUL"(%norm-2, %blk.2.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-2 = linalg.matmul ins(%blk.2.attn_q.weight, %attn_norm-2 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-2 = "MUL_MAT"(%blk.2.attn_q.weight, %attn_norm-2) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-2__reshaped_ = "RESHAPE"(%Qcur-2) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-2 = "ROPE"(%Qcur-2__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-2 = linalg.matmul ins(%blk.2.attn_v.weight, %attn_norm-2 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-2 = "MUL_MAT"(%blk.2.attn_v.weight, %attn_norm-2) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-2 = "RESHAPE"(%Vcur-2) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-2 = linalg.matmul ins(%blk.2.attn_k.weight, %attn_norm-2 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-2 = "MUL_MAT"(%blk.2.attn_k.weight, %attn_norm-2) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-2__reshaped_ = "RESHAPE"(%Kcur-2) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-2 = "ROPE"(%Kcur-2__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-2__view_ = "VIEW"(%Kcur-2) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l2__view_ = "SET_ROWS"(%Kcur-2__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-2__view_ = "VIEW"(%Vcur-2) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l2__view_ = "SET_ROWS"(%Vcur-2__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-2__view_ = "VIEW"(%Qcur-2) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-2__view___permuted_ = "PERMUTE"(%Qcur-2__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l2__view_ = "VIEW"(%cache_k_l2) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l2__view___permuted_ = "PERMUTE"(%cache_k_l2__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l2__view_ = "VIEW"(%cache_v_l2) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l2__view___permuted_ = "PERMUTE"(%cache_v_l2__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__fattn__-2 = "FLASH_ATTN_EXT"(%Qcur-2__view___permuted_, %cache_k_l2__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-2 = "RESHAPE"(%__fattn__-2) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-2 = linalg.matmul ins(%blk.2.attn_output.weight, %kqv_out-2 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-2 = "MUL_MAT"(%blk.2.attn_output.weight, %kqv_out-2) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_inp-2 = linalg.add ins(%attn_out-2, %l_out-1 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_inp-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-2 = "ADD"(%attn_out-2, %l_out-1) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-2 = "RMS_NORM"(%ffn_inp-2) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_norm-2 = "MUL"(%norm-2, %blk.2.ffn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_gate-2_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_gate-2 = linalg.matmul ins(%blk.2.ffn_gate.weight, %ffn_norm-2 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_gate-2_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_gate-2 = "MUL_MAT"(%blk.2.ffn_gate.weight, %ffn_norm-2) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-2_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_up-2 = linalg.matmul ins(%blk.2.ffn_up.weight, %ffn_norm-2 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_up-2_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-2 = "MUL_MAT"(%blk.2.ffn_up.weight, %ffn_norm-2) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_swiglu-2 = "GLU"(%ffn_gate-2, %ffn_up-2) : (tensor<768x3x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_out-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_out-2 = linalg.matmul ins(%blk.2.ffn_down.weight, %ffn_swiglu-2 : tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>)
			outs(%ffn_out-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_out-2 = "MUL_MAT"(%blk.2.ffn_down.weight, %ffn_swiglu-2) : (tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-2_init = tensor.empty() : tensor<288x3x1x1xf32>
		%l_out-2 = linalg.add ins(%ffn_out-2, %ffn_inp-2 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%l_out-2_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-2 = "ADD"(%ffn_out-2, %ffn_inp-2) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-3 = "RMS_NORM"(%l_out-2) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-3 = "MUL"(%norm-3, %blk.3.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-3 = linalg.matmul ins(%blk.3.attn_q.weight, %attn_norm-3 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-3 = "MUL_MAT"(%blk.3.attn_q.weight, %attn_norm-3) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-3__reshaped_ = "RESHAPE"(%Qcur-3) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-3 = "ROPE"(%Qcur-3__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-3 = linalg.matmul ins(%blk.3.attn_v.weight, %attn_norm-3 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-3 = "MUL_MAT"(%blk.3.attn_v.weight, %attn_norm-3) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-3 = "RESHAPE"(%Vcur-3) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-3 = linalg.matmul ins(%blk.3.attn_k.weight, %attn_norm-3 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-3 = "MUL_MAT"(%blk.3.attn_k.weight, %attn_norm-3) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-3__reshaped_ = "RESHAPE"(%Kcur-3) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-3 = "ROPE"(%Kcur-3__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-3__view_ = "VIEW"(%Kcur-3) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l3__view_ = "SET_ROWS"(%Kcur-3__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-3__view_ = "VIEW"(%Vcur-3) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l3__view_ = "SET_ROWS"(%Vcur-3__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-3__view_ = "VIEW"(%Qcur-3) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-3__view___permuted_ = "PERMUTE"(%Qcur-3__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l3__view_ = "VIEW"(%cache_k_l3) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l3__view___permuted_ = "PERMUTE"(%cache_k_l3__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l3__view_ = "VIEW"(%cache_v_l3) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l3__view___permuted_ = "PERMUTE"(%cache_v_l3__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__fattn__-3 = "FLASH_ATTN_EXT"(%Qcur-3__view___permuted_, %cache_k_l3__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-3 = "RESHAPE"(%__fattn__-3) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-3 = linalg.matmul ins(%blk.3.attn_output.weight, %kqv_out-3 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-3 = "MUL_MAT"(%blk.3.attn_output.weight, %kqv_out-3) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_inp-3 = linalg.add ins(%attn_out-3, %l_out-2 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_inp-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-3 = "ADD"(%attn_out-3, %l_out-2) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-3 = "RMS_NORM"(%ffn_inp-3) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_norm-3 = "MUL"(%norm-3, %blk.3.ffn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_gate-3_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_gate-3 = linalg.matmul ins(%blk.3.ffn_gate.weight, %ffn_norm-3 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_gate-3_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_gate-3 = "MUL_MAT"(%blk.3.ffn_gate.weight, %ffn_norm-3) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-3_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_up-3 = linalg.matmul ins(%blk.3.ffn_up.weight, %ffn_norm-3 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_up-3_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-3 = "MUL_MAT"(%blk.3.ffn_up.weight, %ffn_norm-3) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_swiglu-3 = "GLU"(%ffn_gate-3, %ffn_up-3) : (tensor<768x3x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_out-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_out-3 = linalg.matmul ins(%blk.3.ffn_down.weight, %ffn_swiglu-3 : tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>)
			outs(%ffn_out-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_out-3 = "MUL_MAT"(%blk.3.ffn_down.weight, %ffn_swiglu-3) : (tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-3_init = tensor.empty() : tensor<288x3x1x1xf32>
		%l_out-3 = linalg.add ins(%ffn_out-3, %ffn_inp-3 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%l_out-3_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-3 = "ADD"(%ffn_out-3, %ffn_inp-3) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-4 = "RMS_NORM"(%l_out-3) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-4 = "MUL"(%norm-4, %blk.4.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-4 = linalg.matmul ins(%blk.4.attn_q.weight, %attn_norm-4 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-4 = "MUL_MAT"(%blk.4.attn_q.weight, %attn_norm-4) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-4__reshaped_ = "RESHAPE"(%Qcur-4) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-4 = "ROPE"(%Qcur-4__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-4 = linalg.matmul ins(%blk.4.attn_v.weight, %attn_norm-4 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-4 = "MUL_MAT"(%blk.4.attn_v.weight, %attn_norm-4) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-4 = "RESHAPE"(%Vcur-4) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-4 = linalg.matmul ins(%blk.4.attn_k.weight, %attn_norm-4 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-4 = "MUL_MAT"(%blk.4.attn_k.weight, %attn_norm-4) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-4__reshaped_ = "RESHAPE"(%Kcur-4) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-4 = "ROPE"(%Kcur-4__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-4__view_ = "VIEW"(%Kcur-4) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l4__view_ = "SET_ROWS"(%Kcur-4__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-4__view_ = "VIEW"(%Vcur-4) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l4__view_ = "SET_ROWS"(%Vcur-4__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-4__view_ = "VIEW"(%Qcur-4) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-4__view___permuted_ = "PERMUTE"(%Qcur-4__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l4__view_ = "VIEW"(%cache_k_l4) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l4__view___permuted_ = "PERMUTE"(%cache_k_l4__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l4__view_ = "VIEW"(%cache_v_l4) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l4__view___permuted_ = "PERMUTE"(%cache_v_l4__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__fattn__-4 = "FLASH_ATTN_EXT"(%Qcur-4__view___permuted_, %cache_k_l4__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-4 = "RESHAPE"(%__fattn__-4) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-4 = linalg.matmul ins(%blk.4.attn_output.weight, %kqv_out-4 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-4 = "MUL_MAT"(%blk.4.attn_output.weight, %kqv_out-4) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_inp-4 = linalg.add ins(%attn_out-4, %l_out-3 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_inp-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_inp-4 = "ADD"(%attn_out-4, %l_out-3) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-4 = "RMS_NORM"(%ffn_inp-4) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_norm-4 = "MUL"(%norm-4, %blk.4.ffn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_gate-4_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_gate-4 = linalg.matmul ins(%blk.4.ffn_gate.weight, %ffn_norm-4 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_gate-4_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_gate-4 = "MUL_MAT"(%blk.4.ffn_gate.weight, %ffn_norm-4) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-4_init = tensor.empty() : tensor<768x3x1x1xf32>
		%ffn_up-4 = linalg.matmul ins(%blk.4.ffn_up.weight, %ffn_norm-4 : tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%ffn_up-4_init : tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_up-4 = "MUL_MAT"(%blk.4.ffn_up.weight, %ffn_norm-4) : (tensor<288x768x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_swiglu-4 = "GLU"(%ffn_gate-4, %ffn_up-4) : (tensor<768x3x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<768x3x1x1xf32>

		%ffn_out-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%ffn_out-4 = linalg.matmul ins(%blk.4.ffn_down.weight, %ffn_swiglu-4 : tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>)
			outs(%ffn_out-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%ffn_out-4 = "MUL_MAT"(%blk.4.ffn_down.weight, %ffn_swiglu-4) : (tensor<768x288x1x1xf32>, tensor<768x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-4_init = tensor.empty() : tensor<288x3x1x1xf32>
		%l_out-4 = linalg.add ins(%ffn_out-4, %ffn_inp-4 : tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%l_out-4_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%l_out-4 = "ADD"(%ffn_out-4, %ffn_inp-4) : (tensor<288x3x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%norm-5 = "RMS_NORM"(%l_out-4) : (tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_norm-5 = "MUL"(%norm-5, %blk.5.attn_norm.weight) : (tensor<288x3x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-5_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Qcur-5 = linalg.matmul ins(%blk.5.attn_q.weight, %attn_norm-5 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Qcur-5_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-5 = "MUL_MAT"(%blk.5.attn_q.weight, %attn_norm-5) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Qcur-5__reshaped_ = "RESHAPE"(%Qcur-5) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-5 = "ROPE"(%Qcur-5__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Vcur-5_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Vcur-5 = linalg.matmul ins(%blk.5.attn_v.weight, %attn_norm-5 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Vcur-5_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-5 = "MUL_MAT"(%blk.5.attn_v.weight, %attn_norm-5) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Vcur-5 = "RESHAPE"(%Vcur-5) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-5_init = tensor.empty() : tensor<288x3x1x1xf32>
		%Kcur-5 = linalg.matmul ins(%blk.5.attn_k.weight, %attn_norm-5 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%Kcur-5_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-5 = "MUL_MAT"(%blk.5.attn_k.weight, %attn_norm-5) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%Kcur-5__reshaped_ = "RESHAPE"(%Kcur-5) : (tensor<288x3x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-5 = "ROPE"(%Kcur-5__reshaped_, %leaf_4) : (tensor<48x6x3x1xf32>, tensor<3x1x1x1xf32>) -> tensor<48x6x3x1xf32>

		%Kcur-5__view_ = "VIEW"(%Kcur-5) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_k_l5__view_ = "SET_ROWS"(%Kcur-5__view_, %leaf_7) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Vcur-5__view_ = "VIEW"(%Vcur-5) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%cache_v_l5__view_ = "SET_ROWS"(%Vcur-5__view_, %leaf_9) : (tensor<288x3x1x1xf32>, tensor<3x1x1x1xf32>) -> tensor<288x512x1x1xf32>

		%Qcur-5__view_ = "VIEW"(%Qcur-5) : (tensor<48x6x3x1xf32>) -> tensor<48x6x3x1xf32>

		%Qcur-5__view___permuted_ = "PERMUTE"(%Qcur-5__view_) : (tensor<48x6x3x1xf32>) -> tensor<48x3x6x1xf32>

		%cache_k_l5__view_ = "VIEW"(%cache_k_l5) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_k_l5__view___permuted_ = "PERMUTE"(%cache_k_l5__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%cache_v_l5__view_ = "VIEW"(%cache_v_l5) : (tensor<288x512x1x1xf32>) -> tensor<48x6x256x1xf32>

		%cache_v_l5__view___permuted_ = "PERMUTE"(%cache_v_l5__view_) : (tensor<48x6x256x1xf32>) -> tensor<48x256x6x1xf32>

		%__fattn__-5 = "FLASH_ATTN_EXT"(%Qcur-5__view___permuted_, %cache_k_l5__view___permuted_) : (tensor<48x3x6x1xf32>, tensor<48x256x6x1xf32>) -> tensor<48x6x3x1xf32>

		%kqv_out-5 = "RESHAPE"(%__fattn__-5) : (tensor<48x6x3x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-5_init = tensor.empty() : tensor<288x3x1x1xf32>
		%attn_out-5 = linalg.matmul ins(%blk.5.attn_output.weight, %kqv_out-5 : tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>)
			outs(%attn_out-5_init : tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%attn_out-5 = "MUL_MAT"(%blk.5.attn_output.weight, %kqv_out-5) : (tensor<288x288x1x1xf32>, tensor<288x3x1x1xf32>) -> tensor<288x3x1x1xf32>

		%node_180 = "GET_ROWS"(%attn_out-5, %leaf_71) : (tensor<288x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%node_181 = "GET_ROWS"(%l_out-4, %leaf_71) : (tensor<288x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%ffn_inp-5_init = tensor.empty() : tensor<288x1x1x1xf32>
		%ffn_inp-5 = linalg.add ins(%node_180, %node_181 : tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>)
			outs(%ffn_inp-5_init : tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%ffn_inp-5 = "ADD"(%node_180, %node_181) : (tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%norm-5 = "RMS_NORM"(%ffn_inp-5) : (tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%ffn_norm-5 = "MUL"(%norm-5, %blk.5.ffn_norm.weight) : (tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%ffn_gate-5_init = tensor.empty() : tensor<768x1x1x1xf32>
		%ffn_gate-5 = linalg.matmul ins(%blk.5.ffn_gate.weight, %ffn_norm-5 : tensor<288x768x1x1xf32>, tensor<288x1x1x1xf32>)
			outs(%ffn_gate-5_init : tensor<768x1x1x1xf32>) -> tensor<768x1x1x1xf32>

		%ffn_gate-5 = "MUL_MAT"(%blk.5.ffn_gate.weight, %ffn_norm-5) : (tensor<288x768x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<768x1x1x1xf32>

		%ffn_up-5_init = tensor.empty() : tensor<768x1x1x1xf32>
		%ffn_up-5 = linalg.matmul ins(%blk.5.ffn_up.weight, %ffn_norm-5 : tensor<288x768x1x1xf32>, tensor<288x1x1x1xf32>)
			outs(%ffn_up-5_init : tensor<768x1x1x1xf32>) -> tensor<768x1x1x1xf32>

		%ffn_up-5 = "MUL_MAT"(%blk.5.ffn_up.weight, %ffn_norm-5) : (tensor<288x768x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<768x1x1x1xf32>

		%ffn_swiglu-5 = "GLU"(%ffn_gate-5, %ffn_up-5) : (tensor<768x1x1x1xf32>, tensor<768x1x1x1xf32>) -> tensor<768x1x1x1xf32>

		%ffn_out-5_init = tensor.empty() : tensor<288x1x1x1xf32>
		%ffn_out-5 = linalg.matmul ins(%blk.5.ffn_down.weight, %ffn_swiglu-5 : tensor<768x288x1x1xf32>, tensor<768x1x1x1xf32>)
			outs(%ffn_out-5_init : tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%ffn_out-5 = "MUL_MAT"(%blk.5.ffn_down.weight, %ffn_swiglu-5) : (tensor<768x288x1x1xf32>, tensor<768x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%l_out-5_init = tensor.empty() : tensor<288x1x1x1xf32>
		%l_out-5 = linalg.add ins(%ffn_out-5, %ffn_inp-5 : tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>)
			outs(%l_out-5_init : tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%l_out-5 = "ADD"(%ffn_out-5, %ffn_inp-5) : (tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%norm = "RMS_NORM"(%l_out-5) : (tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%result_norm = "MUL"(%norm, %output_norm.weight) : (tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<288x1x1x1xf32>

		%result_output_init = tensor.empty() : tensor<32000x1x1x1xf32>
		%result_output = linalg.matmul ins(%output.weight, %result_norm : tensor<288x32000x1x1xf32>, tensor<288x1x1x1xf32>)
			outs(%result_output_init : tensor<32000x1x1x1xf32>) -> tensor<32000x1x1x1xf32>

		%result_output = "MUL_MAT"(%output.weight, %result_norm) : (tensor<288x32000x1x1xf32>, tensor<288x1x1x1xf32>) -> tensor<32000x1x1x1xf32>

	}
}
