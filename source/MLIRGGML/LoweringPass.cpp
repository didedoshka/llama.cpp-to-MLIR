#include "GGMLOps.h"
#include "Passes.h"

#include "GGMLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;

namespace {
struct LoweringPass : public PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringPass)
    StringRef getArgument() const override { return "ggml_lowering"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<tensor::TensorDialect, arith::ArithDialect, linalg::LinalgDialect>();
    }
    void runOnOperation() final;
};

// template <typename BinaryOp, typename LoweredBinaryOp>
struct AddOpLowering : public OpConversionPattern<ggml::AddOp> {
    using OpConversionPattern<ggml::AddOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<ggml::AddOp>::OpAdaptor;

    LogicalResult matchAndRewrite(ggml::AddOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto initTensor =
            tensor::EmptyOp::create(rewriter, rewriter.getUnknownLoc(), op.getLhs().getType().getShape(),
                                    op.getLhs().getType().getElementType());
        auto addOp = linalg::AddOp::create(rewriter, rewriter.getUnknownLoc(), {op.getLhs(), op.getRhs()},
                                           {initTensor});

        rewriter.replaceOp(op, addOp);
        return success();
    }
};
// using AddOpLowering = BinaryOpLowering<ggml::AddOp, linalg::AddOp>;

// https://discourse.llvm.org/t/lowering-of-scatter-operations/70535
// tensor.gather cannot be lowered in any way
struct GetRowsOpLowering : public OpConversionPattern<ggml::GetRowsOp> {
    using OpConversionPattern<ggml::GetRowsOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<ggml::GetRowsOp>::OpAdaptor;

    LogicalResult matchAndRewrite(ggml::GetRowsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto zero = arith::ConstantOp::create(rewriter, rewriter.getUnknownLoc(), rewriter.getIndexAttr(0));
        auto one = arith::ConstantOp::create(rewriter, rewriter.getUnknownLoc(), rewriter.getIndexAttr(1));
        auto nInt = op.getIndices().getType().getShape()[3];
        auto n = arith::ConstantOp::create(rewriter, rewriter.getUnknownLoc(), rewriter.getIndexAttr(nInt));
        auto initTensor =
            tensor::EmptyOp::create(rewriter, rewriter.getUnknownLoc(), op.getResult().getType().getShape(),
                                    op.getResult().getType().getElementType());

        auto loop = scf::ForOp::create(rewriter, rewriter.getUnknownLoc(), zero, n, one, {initTensor});
        // auto loop = scf::ForOp::create(rewriter, rewriter.getUnknownLoc(), zero, n, one);
        auto tensorPrev = loop.getRegionIterArg(0);
        rewriter.setInsertionPointToStart(loop.getBody());
        auto i = loop.getInductionVar();
        auto j = tensor::ExtractOp::create(rewriter, rewriter.getUnknownLoc(), op.getIndices(),
                                           {zero, zero, zero, i});
        auto jIndex = index::CastSOp::create(rewriter, rewriter.getUnknownLoc(), rewriter.getIndexType(), j);

        auto mInt = op.getValues().getType().getShape()[3];
        auto mAttr = rewriter.getIndexAttr(mInt);
        auto zeroAttr = rewriter.getIndexAttr(0);
        auto oneAttr = rewriter.getIndexAttr(1);
        auto slice = tensor::ExtractSliceOp::create(
            rewriter, rewriter.getUnknownLoc(), op.getValues(),
            {OpFoldResult(zeroAttr), OpFoldResult(zeroAttr), OpFoldResult(jIndex), OpFoldResult(zeroAttr)},
            {OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(mAttr)},
            {OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr)});
        auto tensorCur = tensor::InsertSliceOp::create(
            rewriter, rewriter.getUnknownLoc(), slice, tensorPrev,
            {OpFoldResult(zeroAttr), OpFoldResult(zeroAttr), OpFoldResult(i), OpFoldResult(zeroAttr)},
            {OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(mAttr)},
            {OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr), OpFoldResult(oneAttr)});
        scf::YieldOp::create(rewriter, rewriter.getUnknownLoc(), {tensorCur});

        rewriter.replaceOp(op, loop);
        return success();
    }
};

tensor::ReshapeOp reshape2DTo4D(ConversionPatternRewriter &rewriter, TypedValue<TensorType> tensor) {
    auto tensorShape = tensor.getType().getShape();
    auto resultType =
        RankedTensorType::get({1ll, 1ll, tensorShape[0], tensorShape[1]}, tensor.getType().getElementType());
    auto resultShape = mlir::arith::ConstantOp::create(
        rewriter, rewriter.getUnknownLoc(),
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get({4}, rewriter.getIndexType()),
                                     {1ll, 1ll, tensorShape[0], tensorShape[1]}));
    auto result =
        tensor::ReshapeOp::create(rewriter, rewriter.getUnknownLoc(), resultType, tensor, resultShape);
    return result;
}

tensor::ReshapeOp reshape4DTo2D(ConversionPatternRewriter &rewriter, TypedValue<TensorType> tensor) {
    auto tensorShape = tensor.getType().getShape();
    auto resultType =
        RankedTensorType::get({tensorShape[2], tensorShape[3]}, tensor.getType().getElementType());
    auto resultShape = mlir::arith::ConstantOp::create(
        rewriter, rewriter.getUnknownLoc(),
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get({2}, rewriter.getIndexType()),
                                     {tensorShape[2], tensorShape[3]}));
    auto result =
        tensor::ReshapeOp::create(rewriter, rewriter.getUnknownLoc(), resultType, tensor, resultShape);
    return result;
}

struct MulMatOpLowering : public OpConversionPattern<ggml::MulMatOp> {
    using OpConversionPattern<ggml::MulMatOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<ggml::MulMatOp>::OpAdaptor;

    LogicalResult matchAndRewrite(ggml::MulMatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto a = reshape4DTo2D(rewriter, op.getA());
        auto b = reshape4DTo2D(rewriter, op.getB());
        auto bShape = b.getType().getShape();

        auto bTransposedShape = {bShape[1], bShape[0]};
        auto initBTransposed = tensor::EmptyOp::create(rewriter, rewriter.getUnknownLoc(), bTransposedShape,
                                                       b.getType().getElementType());
        auto bTransposed =
            linalg::TransposeOp::create(rewriter, rewriter.getUnknownLoc(), b, initBTransposed, {1, 0});

        auto initResultTensor = tensor::EmptyOp::create(rewriter, rewriter.getUnknownLoc(),
                                                        {bShape[0], a.getType().getShape()[0]},
                                                        op.getResult().getType().getElementType());

        SmallVector<Value> matmulInputs{bTransposed.getResult()};
        matmulInputs.insert(matmulInputs.begin(), a);
        auto result =
            linalg::MatmulOp::create(rewriter, rewriter.getUnknownLoc(), matmulInputs, {initResultTensor});

        auto result4D = reshape2DTo4D(rewriter, cast<TypedValue<TensorType>>(result->getResult(0)));

        rewriter.replaceOp(op, result4D);
        return success();
    }
};
} // namespace

void LoweringPass::runOnOperation() {
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
    target.addLegalDialect<tensor::TensorDialect, arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                           index::IndexDialect>();
    target.addIllegalDialect<ggml::GGMLDialect>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering>(&getContext());
    patterns.add<GetRowsOpLowering>(&getContext());
    patterns.add<MulMatOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::ggml::createLoweringPass() { return std::make_unique<LoweringPass>(); }
