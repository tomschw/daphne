/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <parser/sql/morphstore/MorphStoreSQLVisitor.h>
#include <parser/sql/SQLVisitor.cpp>

antlrcpp::Any MorphStoreSQLVisitor::visitCmpExpr(
        SQLGrammarParser::CmpExprContext * ctx
)
{
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(vLhs);
    mlir::Value rhs = utils.valueOrError(vRhs);

    if(op == "=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectEqOp>(
                loc, lhs, rhs
        ));
    if(op == "<>")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectNeqOp>(
                loc, lhs, rhs
        ));
    if(op == "<")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectLtOp>(
                loc, lhs, rhs
        ));
    if(op == "<=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectLeOp>(
                loc, lhs, rhs
        ));
    if(op == ">")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectGtOp>(
                loc, lhs, rhs
        ));
    if(op == ">=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectGeOp>(
                loc, lhs, rhs
        ));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any MorphStoreSQLVisitor::visitWhereClause(
    SQLGrammarParser::WhereClauseContext * ctx
)
{
    //Creates a FilterRowOp with the result of a generalExpr. The result is a
    //matrix or a single value, vExpr. vExpr gets cast to a matrix, which
    //FilterRowOp uses. IMPORTANT: FilterRowOp takes up the work to make a
    //int/float into a boolean for the filtering.
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value filter;

    antlrcpp::Any vExpr = visit(ctx->cond);
    mlir::Value expr = utils.valueOrError(vExpr);
    filter = castToMatrixColumn(expr);

    mlir::Value v = static_cast<mlir::Value>(
        builder.create<mlir::daphne::ProjectOp>(
            loc,
            currentFrame.getType(),
            currentFrame,
            filter
        )
    );
    return v;
}

//havingClause
antlrcpp::Any MorphStoreSQLVisitor::visitHavingClause(
    SQLGrammarParser::HavingClauseContext * ctx
)
{
    //Same as Where
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value filter;
    antlrcpp::Any vExpr = visit(ctx->cond);

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value expr = utils.valueOrError(vExpr);
    filter = castToMatrixColumn(expr);

    mlir::Value v = static_cast<mlir::Value>(
        builder.create<mlir::daphne::ProjectOp>(
            loc,
            currentFrame.getType(),
            currentFrame,
            filter
        )
    );
    return v;
}

antlrcpp::Any MorphStoreSQLVisitor::visitMulExpr(
    SQLGrammarParser::MulExprContext * ctx
)
{
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(vLhs);
    mlir::Value rhs = utils.valueOrError(vRhs);

    if(op == "*")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphMulOp>(
            loc, lhs, rhs
        ));
    if(op == "/")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphDivOp>(
            loc, lhs, rhs
        ));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any MorphStoreSQLVisitor::visitAddExpr(
    SQLGrammarParser::AddExprContext * ctx
)
{
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(vLhs);
    mlir::Value rhs = utils.valueOrError(vRhs);

    if(op == "+")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphAddOp>(
            loc, lhs, rhs
        ));
    if(op == "-")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphSubOp>(
            loc, lhs, rhs
        ));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any MorphStoreSQLVisitor::visitGroupAggExpr(
    SQLGrammarParser::GroupAggExprContext * ctx
)
{
    //This function should only be called if there is a "group by" in the query
    //Codegeneration = false:
    //  This function activates the code generation and ignores the aggreagtion
    //  It lets the generalExpr create code as usual. When a Value is returned
    //  it takes this value and adds it to the currentFrame under a new and
    //  somewhat unique name.
    //  (TODO: there might be an issue with not unique name generation)
    //  the name gets saved alongside with the aggregation function.
    //  After all that, code generation is turned off again.
    //Codegeneration = true:
    //  The function looks up the unique name again and extracts a matrix from
    //  the currentFrame. This Matrix is the result of this function.
    std::string newColumnName = "group_" + ctx->var->getText();

    // Run aggreagation for whole column
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::group) && isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){  
        mlir::Location loc = utils.getLoc(ctx->start);

        mlir::Value col = utils.valueOrError(visit(ctx->var));

        mlir::Type resTypeCol = col.getType().dyn_cast<mlir::daphne::MatrixType>().getElementType();

        const std::string &func = ctx->func->getText();

        mlir::Value result; 
        if(func == "count"){
            result = utils.castSI64If(static_cast<mlir::Value>(
            builder.create<mlir::daphne::NumRowsOp>(
                loc,
                utils.sizeType,
                col
            )));
        }
        if(func == "sum"){
            result = static_cast<mlir::Value>(
            builder.create<mlir::daphne::MorphStoreAggSumOp>(
                loc,
                resTypeCol,
                col
                )
            );
        }
        if(func == "min"){
            result = static_cast<mlir::Value>(
            builder.create<mlir::daphne::AllAggMinOp>(
                loc,
                resTypeCol,
                col
                ) 
            );
        }
        if(func == "max"){
            result = static_cast<mlir::Value>(
            builder.create<mlir::daphne::AllAggMaxOp>(
                loc,
                resTypeCol,
                col
                )
            );
        }
        if(func == "avg"){
            result = static_cast<mlir::Value>(
            builder.create<mlir::daphne::AllAggMeanOp>(
                loc,
                resTypeCol,
                col
                )
            );
        }

        std::string newColumnNameAppended = getEnumLabelExt(ctx->func->getText()) + "(" + newColumnName + ")";

        return utils.castIf(utils.matrixOf(result), result);

        std::stringstream x;
        x << "Error: " << func << " does not name a supported aggregation function";
        throw std::runtime_error(x.str());
        
    }
    if(isBitSet(sqlFlag, (int64_t)SQLBit::agg)){ //Not allowed nested Function Call
        throw std::runtime_error("Nested Aggregation Functions");
    }

    //create Column pre Group for in group Aggregation
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        columnName.push_back(createStringConstant(newColumnName));
        functionName.push_back(getGroupEnum(ctx->func->getText()));

        setBit(sqlFlag, (int64_t)SQLBit::agg, 1);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);
        mlir::Value expr = utils.valueOrError(visit(ctx->generalExpr()));
        setBit(sqlFlag, (int64_t)SQLBit::agg, 0);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 0);

        mlir::Value matrix = castToMatrixColumn(expr);
        currentFrame = addMatrixToCurrentFrame(matrix, newColumnName);
        return nullptr;
    }else{ //Get Column after Group
        std::string newColumnNameAppended = getEnumLabelExt(ctx->func->getText()) + "(" + newColumnName + ")";
        mlir::Value colname = utils.valueOrError(createStringConstant(newColumnNameAppended));
        return extractMatrixFromFrame(currentFrame, colname); //returns Matrix
    }
}

//joinExpr
antlrcpp::Any MorphStoreSQLVisitor::visitInnerJoin(
    SQLGrammarParser::InnerJoinContext * ctx
)
{
    //we join to frames together. One is the currentFrame and the other is a
    //new frame. The argument that referneces the currentFrame has to be on the
    //left side of the Comparisons and the to be joined on the right side.
    //This behavior could be changed here.
    //TODO: Make the position independent
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value tojoin = utils.valueOrError(visit(ctx->var));


    std::vector<mlir::Type> colTypes;
    for(mlir::Type t : currentFrame.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    for(mlir::Type t : tojoin.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

//ctx->CMP_OP(0)->getText()
    if(ctx->op->getText() == "=" && ctx->selectIdent().size() == 2){
        //rhs is join
        //lhs is currentFrame
        mlir::Value rhsName = utils.valueOrError(visit(ctx->rhs));
        mlir::Value lhsName = utils.valueOrError(visit(ctx->lhs));

        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::MorphJoinOp>(
                loc,
                t,
                currentFrame,
                tojoin,
                rhsName,
                lhsName
            ));
    }

    std::vector<mlir::Value> rhsNames;
    std::vector<mlir::Value> lhsNames;
    std::vector<mlir::Attribute> ops;

    for(auto i = 0ul; i < ctx->selectIdent().size()/2; i++){
        mlir::Value lhsName = utils.valueOrError(visit(ctx->selectIdent(i*2)));
        mlir::Value rhsName = utils.valueOrError(visit(ctx->selectIdent(i*2 + 1)));
        mlir::Attribute op = getCompareEnum(ctx->CMP_OP(i)->getText());

        lhsNames.push_back(lhsName);
        rhsNames.push_back(rhsName);
        ops.push_back(op);

    }

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::ThetaJoinOp>(
            loc,
            t,
            currentFrame,
            tojoin,
            lhsNames,
            rhsNames,
            builder.getArrayAttr(ops)
        )
    );

}

antlrcpp::Any MorphStoreSQLVisitor::visitAndExpr(
    SQLGrammarParser::AndExprContext * ctx
)
{
    mlir::Location loc = utils.getLoc(ctx->start);

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(vLhs);
    mlir::Value rhs = utils.valueOrError(vRhs);

    lhs = castToIntMatrixColumn(lhs);
    rhs = castToIntMatrixColumn(rhs);

    return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphAndOp>(
        loc, lhs, rhs
    ));
}